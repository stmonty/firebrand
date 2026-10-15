"""Exact parallel evaluation of a compiled Transformer on a GPU.

Given a whole trace, the forward pass at every position reads only earlier
tokens, so all positions run at once. If the argmax at every position equals
the next token, the model generates exactly that trace (see speculative.py).

Values are int64, which is exact because every weight and residual value of a
compiled model is an integer below 2**53. Heads use the same exact indexes as
the compiled decoder. FFN units run in Triton kernels, and units that read one
small-range row become table gathers. Residual slots with disjoint lifetimes
share rows. Positions are grouped by record (the model's phase feature); rows
are checked on the device for being constant within every group, and units
whose inputs all are run once per group.
"""

import math

import numpy as np
import torch

from .runtime import Sparse

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton ships with CUDA builds of PyTorch.
    triton = None

TABLE_SPAN = 1024  # Tabulate single-input units with at most this many input values.
POSITION_BITS = 26  # Positions below 2**25 (the compiler's context limit) fit.
TABLE_KEY_BITS = 24  # Match keys up to this wide use a direct table of newest positions.
ERRORS = ("address outside causal prefix", "no sentinel in causal prefix", "match key digits outside 0..255")


def _coo(weight):
    if isinstance(weight, Sparse):
        rows, cols, vals = weight.rows, weight.cols, weight.vals
    else:
        weight = np.asarray(weight)
        rows, cols = np.nonzero(weight)
        vals = weight[rows, cols]
    if not np.all(vals == np.round(vals)):
        raise ValueError("the GPU verifier needs integer weights")
    order = np.lexsort((rows, cols))
    return rows[order].astype(np.int64), cols[order].astype(np.int64), np.round(vals[order]).astype(np.int64)


if triton is not None:

    @triton.jit
    def _read(F, L, sF, sL, inv, row, offsets, group, mask, MODE: tl.constexpr):
        """Row `row` at a block of positions (MODE 1) or groups (MODE 0)."""
        row = row.to(tl.int64)
        if MODE == 0:
            value = tl.load(L + row * sL + offsets, mask=mask, other=0)
        else:
            if tl.load(inv + row) != 0:
                value = tl.load(L + row * sL + group, mask=mask, other=0)
            else:
                value = tl.load(F + row * sF + offsets, mask=mask, other=0)
        return value

    @triton.jit
    def _unit(
        F,
        L,
        sF,
        sL,
        inv,
        u,
        offsets,
        group,
        mask,
        unit_neurons,
        neuron_entries,
        entry_row,
        entry_val,
        neuron_out,
        BLOCK: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """sum_n out(n) * relu(sum_e val(e) * x[row(e)]) for unit u."""
        n0 = tl.load(unit_neurons + u)
        n1 = tl.load(unit_neurons + u + 1)
        out = tl.zeros([BLOCK], dtype=tl.int64)
        for n in range(n0, n1):
            e0 = tl.load(neuron_entries + n)
            e1 = tl.load(neuron_entries + n + 1)
            acc = tl.zeros([BLOCK], dtype=tl.int64)
            for e in range(e0, e1):
                x = _read(F, L, sF, sL, inv, tl.load(entry_row + e), offsets, group, mask, MODE)
                acc += x * tl.load(entry_val + e)
            out += tl.maximum(acc, 0) * tl.load(neuron_out + n)
        return out

    @triton.jit
    def _ffn_kernel(
        F,
        L,
        sF,
        sL,
        inv,
        group_of,
        n,
        ids,
        unit_slot,
        unit_neurons,
        neuron_entries,
        entry_row,
        entry_val,
        neuron_out,
        table_row,
        table_low,
        table_start,
        table_span,
        tables,
        BLOCK: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """Run units ids[program 0] at a block of positions (MODE 1, into F) or
        groups (MODE 0, into L)."""
        u = tl.load(ids + tl.program_id(0))
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        group = offsets
        if MODE == 1:
            group = tl.load(group_of + offsets, mask=mask, other=0)
        row = tl.load(table_row + u)
        if row >= 0:
            index = _read(F, L, sF, sL, inv, row, offsets, group, mask, MODE) - tl.load(table_low + u)
            inside = (index >= 0) & (index < tl.load(table_span + u))
            out = tl.load(tables + tl.load(table_start + u) + index, mask=mask & inside, other=0)
            if tl.sum((mask & ~inside).to(tl.int32)) > 0:  # Outside the table: compute.
                direct = _unit(
                    F,
                    L,
                    sF,
                    sL,
                    inv,
                    u,
                    offsets,
                    group,
                    mask,
                    unit_neurons,
                    neuron_entries,
                    entry_row,
                    entry_val,
                    neuron_out,
                    BLOCK,
                    MODE,
                )
                out = tl.where(inside, out, direct)
        else:
            out = _unit(
                F,
                L,
                sF,
                sL,
                inv,
                u,
                offsets,
                group,
                mask,
                unit_neurons,
                neuron_entries,
                entry_row,
                entry_val,
                neuron_out,
                BLOCK,
                MODE,
            )
        slot = tl.load(unit_slot + u).to(tl.int64)
        if MODE == 0:
            tl.store(L + slot * sL + offsets, out, mask=mask)
        else:
            tl.store(F + slot * sF + offsets, out, mask=mask)

    @triton.jit
    def _differs_kernel(F, sF, rows, group_of, leaders, n, flags, BLOCK: tl.constexpr):
        """flags[i] = 1 if row rows[i] differs anywhere from its group leader's value."""
        i = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        row = tl.load(rows + i).to(tl.int64)
        leader = tl.load(leaders + tl.load(group_of + offsets, mask=mask, other=0), mask=mask, other=0)
        here = tl.load(F + row * sF + offsets, mask=mask, other=0)
        there = tl.load(F + row * sF + leader, mask=mask, other=0)
        if tl.sum((here != there).to(tl.int32)) > 0:
            tl.atomic_max(flags + i, 1)

    @triton.jit
    def _leaders_kernel(F, L, sF, sL, rows, leaders, groups, BLOCK: tl.constexpr):
        """L[row, g] = F[row, leaders[g]] for each row of rows."""
        row = tl.load(rows + tl.program_id(0)).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < groups
        leader = tl.load(leaders + offsets, mask=mask, other=0)
        tl.store(L + row * sL + offsets, tl.load(F + row * sF + leader, mask=mask, other=0), mask=mask)

    @triton.jit
    def _absolute_kernel(
        F,
        L,
        sF,
        sL,
        inv,
        group_of,
        n,
        positions,
        tokens,
        embedding,
        embedding_columns,
        lower,
        out,
        s_out,
        errors,
        q_start,
        q_row,
        q_val,
        v_start,
        v_col,
        v_val,
        BLOCK: tl.constexpr,
    ):
        """Absolute heads of one stage: the value of the input at the address each
        query names (query coordinate 0), for head program 0."""
        h = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        group = tl.load(group_of + offsets, mask=mask, other=0)
        address = tl.zeros([BLOCK], dtype=tl.int64)
        for e in range(tl.load(q_start + h), tl.load(q_start + h + 1)):
            x = _read(F, L, sF, sL, inv, tl.load(q_row + e), offsets, group, mask, 1)
            address += x * tl.load(q_val + e)
        position = tl.load(positions + offsets, mask=mask, other=0)
        bad = mask & ((address < 0) | (address > position))
        tl.atomic_add(errors + 0, tl.sum(bad.to(tl.int32)))
        address = tl.where(bad, 0, address)
        token = tl.load(tokens + address, mask=mask, other=lower)
        value = tl.zeros([BLOCK], dtype=tl.int64)
        for e in range(tl.load(v_start + h), tl.load(v_start + h + 1)):
            column = tl.load(v_col + e)
            if column == 0:
                x = tl.full([BLOCK], 1, dtype=tl.int64)
            elif column == 1:
                x = address
            elif column == 2:
                x = address * address
            else:
                x = tl.load(embedding + (token - lower) * embedding_columns + column, mask=mask, other=0)
            value += x * tl.load(v_val + e)
        tl.store(out + h * s_out + offsets, value, mask=mask)


def _tabulate(unit, row, low, high):
    """The unit's output for every integer value of its one input row (row 0 is
    the constant one), computed exactly in int64."""
    values = np.arange(low, high + 1, dtype=np.int64)
    out = np.zeros(len(values), dtype=np.int64)
    for neuron_rows, neuron_vals, neuron_out in unit["neurons"]:
        acc = np.zeros(len(values), dtype=np.int64)
        for r, v in zip(neuron_rows, neuron_vals):
            acc += v * (values if r == row else 1)
        out += np.maximum(acc, 0) * neuron_out
    return out


def _split(w_in, w_out):
    """Units: the neurons writing each output slot, with their input entries."""
    in_rows, in_cols, in_vals = _coo(w_in)
    out_rows, out_cols, out_vals = _coo(w_out)
    if len(np.unique(out_rows)) != len(out_rows):
        raise ValueError("a neuron writes more than one slot")
    by_neuron = {}
    for r, c, v in zip(in_rows.tolist(), in_cols.tolist(), in_vals.tolist()):
        by_neuron.setdefault(c, ([], []))
        by_neuron[c][0].append(r)
        by_neuron[c][1].append(v)
    units = {}
    for n, slot, v in zip(out_rows.tolist(), out_cols.tolist(), out_vals.tolist()):
        rows, vals = by_neuron.get(n, ([], []))
        units.setdefault(slot, []).append((rows, vals, v))
    return [{"slot": slot, "neurons": neurons} for slot, neurons in sorted(units.items())]


class Verifier:
    """Next-token predictions at many positions of a given trace, in parallel."""

    def __init__(self, model, device="cuda", chunk=1 << 19, block=256, share=True, graphs=True):
        if triton is None:
            raise RuntimeError("the GPU verifier needs Triton (part of CUDA builds of PyTorch)")
        c, w = model.config, model.weights
        self.config, self.device, self.chunk, self.block = c, torch.device(device), chunk, block
        self.lower, self.vocab = c["lower"], c["upper"] - c["lower"] + 1
        self.max_context = c["max_context"]
        self.features = c.get("position_features", [])
        self.input_width = 4 + len(self.features)
        embedding = np.asarray(w["embedding"])
        if not np.all(embedding == np.round(embedding)):
            raise ValueError("the GPU verifier needs integer embeddings")
        self.embedding_columns = min(embedding.shape[1], self.input_width)
        self.embedding = torch.tensor(
            np.round(embedding[:, : self.embedding_columns]), dtype=torch.int64, device=self.device
        )
        kinds = c.get("head_kinds", [])
        checked = c.get("addressing") == "absolute_checked"
        self.head_kind = [
            kinds[h] if h < len(kinds) and kinds[h] in ("latest", "match") else ("absolute" if checked else "scan")
            for h in range(c["heads"])
        ]
        self.match = {int(h): meta for h, meta in c.get("match_heads", {}).items()}
        heads = {h: {part: _coo(w[f"head_{h}_{part}"]) for part in "qkvo"} for h in range(c["heads"])}
        from .fast import _row_ranges

        ranges = _row_ranges(c, len(w["embedding"]))
        schedule = c.get("schedule") or (
            [{"kind": "attention", "index": h} for h in range(c["heads"])]
            + [{"kind": "ffn", "index": i} for i in range(c["layers"])]
        )

        # Stages in schedule order, with the logical rows each reads and writes.
        stages = []
        for step in schedule:
            if step["kind"] == "attention":
                group = step.get("heads", [step.get("index")])
                reads = set()
                for h in group:
                    reads |= set(heads[h]["q"][0].tolist())
                    if self.head_kind[h] == "match":
                        reads |= set(heads[h]["k"][0].tolist())
                writes = [int(c_) for h in group for c_ in heads[h]["o"][1].tolist()]
                stages.append(("attention", group, reads, writes))
            else:
                units = _split(w[f"ff_{step['index']}_in"], w[f"ff_{step['index']}_out"])
                reads = {r for u in units for rows, _, _ in u["neurons"] for r in rows}
                stages.append(("ffn", units, reads, [u["slot"] for u in units]))
        out_rows, out_cols, out_vals = _coo(w["unembedding"])
        stages = self._prune(stages, heads, set(out_rows.tolist()))
        self._allocate(c["width"], stages, set(out_rows.tolist()), share)
        phys = self.phys

        # Attention heads, with rows renamed to physical rows.
        self.heads = {}
        for h, parts in heads.items():
            q, k, v, o = parts["q"], parts["k"], parts["v"], parts["o"]
            self.heads[h] = {
                "q": (phys[q[0]], q[1], q[2]),
                "k": (phys[k[0]], k[1], k[2]),
                "v": v,
                "o": (o[1], o[2], phys[o[1]]),
            }
            if len(set(o[1].tolist())) != len(o[1]):
                raise ValueError("a head writes one slot twice")

        # FFN units: one flat set of device arrays; each stage keeps its unit ids
        # and, for planning, the physical rows each unit reads.
        flat = {
            key: []
            for key in ("slot", "neurons", "entries", "rows", "vals", "out", "t_row", "t_low", "t_start", "t_span")
        }
        flat["neurons"], flat["entries"], tables = [0], [0], []
        self.stages = []
        for kind, payload, _, _ in stages:
            if kind == "attention":
                self.stages.append(self._attention_stage(payload))
                continue
            first = len(flat["slot"])
            unit_reads = []
            for unit in payload:
                inputs = sorted({r for rows, _, _ in unit["neurons"] for r in rows} - {0})
                flat["slot"].append(int(phys[unit["slot"]]))
                for rows, vals, out in unit["neurons"]:
                    flat["rows"] += phys[np.array(rows, dtype=np.int64)].tolist()
                    flat["vals"] += vals
                    flat["entries"].append(len(flat["rows"]))
                    flat["out"].append(out)
                flat["neurons"].append(len(flat["out"]))
                table = None
                if len(inputs) == 1 and inputs[0] in ranges and len(unit["neurons"]) >= 2:
                    low, high = ranges[inputs[0]]
                    if high - low < TABLE_SPAN:
                        table = _tabulate(unit, inputs[0], low, high)
                if table is None:
                    flat["t_row"].append(-1), flat["t_low"].append(0), flat["t_start"].append(0)
                    flat["t_span"].append(0)
                else:
                    flat["t_row"].append(int(phys[inputs[0]])), flat["t_low"].append(low)
                    flat["t_start"].append(sum(len(t) for t in tables)), flat["t_span"].append(len(table))
                    tables.append(table)
                unit_reads.append(
                    sorted(
                        set(
                            phys[
                                np.array(sorted({r for rows, _, _ in unit["neurons"] for r in rows}), dtype=np.int64)
                            ].tolist()
                        )
                    )
                )
            ids = np.arange(first, len(flat["slot"]), dtype=np.int32)
            starts = np.cumsum([0] + [len(r) for r in unit_reads])
            self.stages.append(
                {
                    "kind": "ffn",
                    "ids": ids,
                    "reads": np.array(sum(unit_reads, []), dtype=np.int64),
                    "starts": starts,
                    "slots": np.array([flat["slot"][i] for i in ids], dtype=np.int64),
                    "slots_device": torch.tensor([flat["slot"][i] for i in ids], device=self.device),
                    "plans": {},
                }
            )
        as_tensor = lambda values, dtype=torch.int64: torch.tensor(values, dtype=dtype, device=self.device)
        self.units = {key: as_tensor(values or [0]) for key, values in flat.items()}
        self.tables = as_tensor(np.concatenate(tables) if tables else [0])

        # Output: logits from a few rows; tabulated when they are the constant and one row.
        self.output_rows = sorted(set(out_rows.tolist()))
        dense = np.zeros((len(self.output_rows), self.vocab), dtype=np.int64)
        np.add.at(dense, ([self.output_rows.index(r) for r in out_rows], out_cols), out_vals)
        self.unembedding = as_tensor(dense)
        self.output_phys = phys[np.array(self.output_rows, dtype=np.int64)]
        self.output_table = None
        varying = [r for r in self.output_rows if r != 0]
        if len(varying) == 1 and varying[0] in ranges and ranges[varying[0]][1] - ranges[varying[0]][0] < TABLE_SPAN:
            low, high = ranges[varying[0]]
            values = np.arange(low, high + 1, dtype=np.int64)
            logits = np.zeros((len(values), self.vocab), dtype=np.int64)
            for i, r in enumerate(self.output_rows):
                logits += (values[:, None] if r == varying[0] else 1) * dense[i][None, :]
            self.output_table = (int(phys[varying[0]]), low, as_tensor(logits.argmax(axis=1) + self.lower))

        # Grouping: the first periodic phase feature, if any.
        mods = [f for f in self.features if f["kind"] == "mod" and f["period"] > 1]
        self.period = (mods[0]["period"], mods[0]["offset"]) if mods and share else None
        if self.period is not None:  # Full chunks then all have the same groups.
            self.chunk = max(self.period[0], chunk // self.period[0] * self.period[0])
        kept = {h for stage in self.stages if stage["kind"] == "attention" for h in stage["heads"]}
        self.graphs = (
            graphs
            and self.period is not None
            and self.device.type == "cuda"
            and all(
                self.head_kind[h] == "absolute"
                or (self.head_kind[h] == "match" and 8 * self.match[h]["digits"] <= TABLE_KEY_BITS)
                for h in kept
            )
        )

    # -- Setup ---------------------------------------------------------------------

    def _prune(self, stages, heads, output_rows):
        """Drop heads, units, and projection columns that cannot affect a prediction
        when heads are answered by exact indexes (which never read, for example, a
        match head's squared key digits)."""

        def columns(h, part):
            kind = self.head_kind[h]
            if kind == "absolute":
                return {0} if part == "q" else set()
            if kind == "match":
                d = self.match[h]["digits"]
                return set(range(d)) if part == "q" else set(range(d)) | {2 * d, 2 * d + 2}
            return None  # Latest and scanning heads: everything.

        needed, kept = set(output_rows), []
        for kind, payload, _, _ in reversed(stages):
            if kind == "attention":
                group = [h for h in payload if set(heads[h]["o"][1].tolist()) & needed]
                for h in group:
                    for part in "qk":
                        wanted = columns(h, part)
                        rows, cols, vals = heads[h][part]
                        if wanted is not None:
                            keep = np.isin(cols, sorted(wanted))
                            heads[h][part] = rows, cols, vals = rows[keep], cols[keep], vals[keep]
                        if part == "q" or self.head_kind[h] == "match":
                            needed |= set(rows.tolist())
                if group:
                    reads = {
                        r
                        for h in group
                        for part in ("q", "k")
                        if part == "q" or self.head_kind[h] == "match"
                        for r in heads[h][part][0].tolist()
                    }
                    kept.append(("attention", group, reads, [int(c_) for h in group for c_ in heads[h]["o"][1]]))
            else:
                units = [u for u in payload if u["slot"] in needed]
                reads = {r for u in units for rows, _, _ in u["neurons"] for r in rows}
                needed |= reads
                if units:
                    kept.append(("ffn", units, reads, [u["slot"] for u in units]))
        return kept[::-1]

    def _allocate(self, width, stages, output_rows, share):
        """Assign slots to physical rows; slots with disjoint lifetimes share a row.
        Slots read but never written map to a row that stays zero."""
        written, last = {}, {}
        for column in range(self.input_width):
            written[column] = -1
        for s, (_, _, reads, writes) in enumerate(stages):
            for r in reads:
                last[r] = max(last.get(r, -1), s)
            for slot in writes:
                if slot in written:
                    raise ValueError("a slot is written twice")
                written[slot] = s
        for r in output_rows:
            last[r] = len(stages)
        never = sorted(set(last) - set(written))
        phys = np.full(width, -1, dtype=np.int64)
        # Rows 0..input_width-1 hold the inputs; then one zero row for unwritten slots.
        self.zero_row = self.input_width
        for column in range(self.input_width):
            phys[column] = column
        for slot in never:
            phys[slot] = self.zero_row
        free, busy, rows = [], [], self.input_width + 1
        if not share:
            for slot in sorted(s for s in written if s >= self.input_width):
                phys[slot], rows = rows, rows + 1
        else:
            busy = [(last.get(column, -1), column) for column in range(self.input_width)]
            for slot in sorted((s for s in written if s >= self.input_width), key=lambda s: (written[s], s)):
                start = written[slot]
                for end, row in list(busy):
                    if end < start:
                        busy.remove((end, row))
                        if row >= self.input_width + 1:
                            free.append(row)
                if free:
                    phys[slot] = free.pop()
                else:
                    phys[slot], rows = rows, rows + 1
                busy.append((last.get(slot, start), int(phys[slot])))
        self.rows, self.phys = rows, phys

    def _attention_stage(self, group):
        absolute = [h for h in group if self.head_kind[h] == "absolute" and self._fused(h)]
        q_start, q_row, q_val, v_start, v_col, v_val = [0], [], [], [0], [], []
        for h in absolute:
            rows, cols, vals = self.heads[h]["q"]
            q_row += rows[cols == 0].tolist()
            q_val += vals[cols == 0].tolist()
            q_start.append(len(q_row))
            rows, cols, vals = self.heads[h]["v"]
            v_col += rows.tolist()
            v_val += vals.tolist()
            v_start.append(len(v_col))
        as_tensor = lambda values: torch.tensor(values or [0], dtype=torch.int64, device=self.device)
        return {
            "kind": "attention",
            "heads": list(group),
            "absolute": absolute,
            "others": [h for h in group if h not in absolute],
            "arrays": [as_tensor(a) for a in (q_start, q_row, q_val, v_start, v_col, v_val)],
        }

    def _fused(self, h):
        """Whether the kernel can compute this absolute head: its value reads only
        the constant, the position and its square, or embedding columns."""
        rows, cols, _ = self.heads[h]["v"]
        positional = {f["column"] for f in self.features}
        return bool(np.all(cols == 0)) and all(
            r in (0, 1, 2) or (r < self.embedding_columns and r not in positional) for r in rows.tolist()
        )

    # -- Inputs ----------------------------------------------------------------

    def _input(self, column, tokens, positions):
        """Input coordinate `column` (embedding or positional) at `positions`."""
        if column == 1:
            return positions
        if column == 2:
            return positions * positions
        for f in self.features:
            if f["column"] != column:
                continue
            kind = f["kind"]
            if kind == "mod":
                return torch.remainder(positions + f["offset"], f["period"])
            if kind == "tag":
                value = (positions >= f["start"]) & (positions % f["period"] == f["phase"])
                if f.get("end") is not None and f["end"] >= 0:
                    value &= positions < f["end"]
                return value.long()
            if kind == "point":
                return (positions == f["position"]).long()
            if kind == "digit":
                shifted = positions + f["offset"]
                return torch.where(shifted >= 0, (shifted >> (8 * f["power"])) & 255, 0)
            if kind == "quotient":
                shifted = positions + f["offset"]
                return torch.where(shifted >= 0, torch.div(shifted, f["period"], rounding_mode="floor"), -1)
        if column < self.embedding_columns:
            return self.embedding[tokens[positions] - self.lower, column]
        return torch.zeros_like(positions)

    def _project_inputs(self, coo, dims, tokens, positions):
        """Input coordinates at `positions` times a projection that reads only inputs."""
        rows, cols, vals = coo
        out = torch.zeros((dims, len(positions)), dtype=torch.int64, device=self.device)
        cache = {}
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if r not in cache:
                cache[r] = self._input(r, tokens, positions)
            out[c] += cache[r] * v
        return out

    # -- Chunk state -------------------------------------------------------------

    def _row(self, ctx, row):
        """A physical row's values at every position of the chunk."""
        if ctx["inv"][row]:
            return ctx["L"][row, ctx["group"]]
        return ctx["F"][row, : ctx["n"]]

    def _project(self, ctx, coo, dims):
        rows, cols, vals = coo
        out = torch.zeros((dims, ctx["n"]), dtype=torch.int64, device=self.device)
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            out[c] += self._row(ctx, r) * v
        return out

    def _decide(self, ctx, differs, key):
        """Which rows are constant within every group.

        Discovering asks the device and records the answer. Replaying assumes the
        recorded answer and counts violations on the device; a chunk with any is
        redone. After a failed replay, a row stays shared only if both plans agree.
        """
        if ctx["replay"] is None:
            same = (~differs).cpu().numpy()
            hint = ctx["hint"]
            if hint is not None and ctx["step"] < len(hint) and hint[ctx["step"]]["key"] == key:
                same &= hint[ctx["step"]]["same"]
            entry = {"same": same, "device": torch.as_tensor(same, device=self.device), "key": key}
            ctx["record"].append(entry)
        else:
            entry = ctx["replay"][ctx["step"]]
            ctx["violations"] += (differs & entry["device"]).sum()
            ctx["refinements"] += (~differs & ~entry["device"]).sum()
        ctx["step"] += 1
        return entry

    def _publish(self, ctx, rows, values):
        """Write source rows (inputs or attention outputs) for every position, and
        keep each in group form when it is the same throughout every group."""
        values = torch.stack(values)
        if ctx["share"]:
            leaders = values[:, ctx["leaders"]]
            entry = self._decide(ctx, (values != leaders[:, ctx["group"]]).any(dim=1), tuple(rows))
            same = entry["same"]
        else:
            entry, same = None, np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            ctx["inv"][row] = same[i]
            if same[i]:
                ctx["L"][row, : ctx["groups"]] = leaders[i]
            else:
                ctx["F"][row, : ctx["n"]] = values[i]
        if entry is None or "inv" not in entry:
            inv = torch.as_tensor(ctx["inv"], dtype=torch.int8, device=self.device)
            if entry is None:
                ctx["inv_device"].copy_(inv)
                return
            entry["inv"] = inv
        ctx["inv_device"].copy_(entry["inv"])

    # -- Attention ---------------------------------------------------------------

    def _match(self, h, ctx, q, state):
        meta = self.match[h]
        digits = meta["digits"]
        positions = ctx["positions"]
        k = self._project(ctx, self.heads[h]["k"], 2 * digits + 3)
        tagged = k[2 * digits + 2] == 1
        sentinel = k[2 * digits] == 1
        key, key_ok = self._pack(k[:digits].double())
        ctx["errors"][2] += (tagged & ~key_ok).sum().to(torch.int32)
        query, query_ok = self._pack(q[:digits].double() / (2 * meta["scale"]))
        index = state.get(("match", h))
        if index is None:
            size = 1 << TABLE_KEY_BITS if 8 * digits <= TABLE_KEY_BITS else 0
            index = state[("match", h)] = {
                "table": torch.full((size + 1,), -1, dtype=torch.int64, device=self.device) if size else None,
                "keys": torch.empty(0, dtype=torch.int64, device=self.device),
                "sentinel": torch.full((1,), -1, dtype=torch.int64, device=self.device),
            }
        none = torch.iinfo(torch.int64).max
        mine = torch.sort(torch.where(tagged & key_ok, (key << POSITION_BITS) | positions, none)).values
        mask = (1 << POSITION_BITS) - 1

        def newest(keys, wanted):
            found = torch.searchsorted(keys, (wanted << POSITION_BITS) | positions, right=True) - 1
            candidate = keys[found.clamp(min=0)] if len(keys) else torch.zeros_like(found)
            hit = (found >= 0) & ((candidate >> POSITION_BITS) == wanted) & query_ok
            return torch.where(hit, candidate & mask, -1)

        winner = newest(mine, query)
        if index["table"] is not None:
            earlier = index["table"][torch.where(query_ok, query, len(index["table"]) - 1)]
        else:
            earlier = newest(index["keys"], query)
        winner = torch.where(winner >= 0, winner, earlier)
        # Newest sentinel at or before each position, via a running count of sentinels.
        count = torch.cumsum(sentinel, 0)
        found = torch.full((len(positions) + 2,), -1, dtype=torch.int64, device=self.device)
        found[0] = index["sentinel"][0]
        found.scatter_(0, torch.where(sentinel, count, len(positions) + 1), positions)  # Last: unused.
        newest_sentinel = found[count]
        ctx["errors"][1] += ((winner < 0) & (newest_sentinel < 0)).sum().to(torch.int32)
        winner = torch.where(winner >= 0, winner, newest_sentinel.clamp(min=0))

        def register():
            """Add this chunk's keys for later chunks, once the chunk is accepted."""
            if index["table"] is not None:
                slot = torch.where(tagged & key_ok, key, len(index["table"]) - 1)
                index["table"].scatter_reduce_(0, slot, torch.where(tagged & key_ok, positions, -1), "amax")
            else:
                index["keys"] = torch.sort(torch.cat([index["keys"], mine[mine != none]])).values
            index["sentinel"].copy_(newest_sentinel[-1:])

        ctx["commit"].append(register)
        return winner

    @staticmethod
    def _pack(digits):
        """Key digits (float64, truncated as the compiled decoder does) packed into
        one integer, and whether every digit is within 0..255."""
        digits = digits.trunc().long()
        ok = ((digits >= 0) & (digits <= 255)).all(dim=0)
        key = torch.zeros(digits.shape[1], dtype=torch.int64, device=digits.device)
        for i in range(digits.shape[0]):
            key |= (digits[i] & 255) << (8 * i)
        return key, ok

    def _latest(self, h, q, positions, tokens, state):
        """Nearest tagged value to the query, then the newest such position."""
        if ("latest", h) not in state:
            everything = torch.arange(len(tokens), device=self.device)
            k = self._project_inputs(self.heads[h]["k"], 4, tokens, everything)
            tagged = k[3] == 1
            value = k[0] - self.lower
            state[("latest", h)] = torch.sort((value[tagged] << POSITION_BITS) | everything[tagged]).values
        index = state[("latest", h)]
        ratio = q[0].double() / (-2 * q[1].double())
        if bool((ratio != ratio.floor()).any()):
            return self._scan(h, q, positions, tokens)
        wanted = ratio.long() - self.lower
        if not len(index):
            raise ValueError("no tagged key in causal prefix")

        def newest(v):
            found = torch.searchsorted(index, (v << POSITION_BITS) | positions, right=True) - 1
            candidate = index[found.clamp(min=0)]
            ok = (found >= 0) & ((candidate >> POSITION_BITS) == v) & (v >= 0) & (v < self.vocab)
            return torch.where(ok, candidate & ((1 << POSITION_BITS) - 1), -1)

        winner = newest(wanted)
        for distance in range(1, self.vocab + 1):
            open_ = winner < 0
            if not bool(open_.any()):
                break
            below, above = newest(wanted - distance), newest(wanted + distance)
            winner = torch.where(open_, torch.maximum(below, above), winner)
        if bool((winner < 0).any()):
            raise ValueError("no tagged key in causal prefix")
        return winner

    def _scan(self, h, q, positions, tokens):
        """Score every earlier position (first maximum), for heads without an index."""
        end = int(positions.max()) + 1
        if end * len(positions) > 1 << 28:
            raise ValueError("a scanning head needs a small context on the GPU")
        everything = torch.arange(end, device=self.device)
        k = self._project_inputs(self.heads[h]["k"], q.shape[0], tokens, everything)
        scores = q.double().t() @ k.double()
        scores[everything[None, :] > positions[:, None]] = -math.inf
        winner = scores.argmax(dim=1)
        if self.head_kind[h] == "latest" and bool((k[3, winner] != 1).any()):
            raise ValueError("no tagged key in causal prefix")
        return winner

    def _attention(self, stage, ctx, tokens, state):
        n, values = ctx["n"], {}
        if stage["absolute"]:
            out = torch.empty((len(stage["absolute"]), n), dtype=torch.int64, device=self.device)
            _absolute_kernel[(len(stage["absolute"]), triton.cdiv(n, self.block))](
                ctx["F"],
                ctx["L"],
                ctx["F"].stride(0),
                ctx["L"].stride(0),
                ctx["inv_device"],
                ctx["group"],
                n,
                ctx["positions"],
                tokens,
                self.embedding,
                self.embedding_columns,
                self.lower,
                out,
                out.stride(0),
                ctx["errors"],
                *stage["arrays"],
                BLOCK=self.block,
            )
            for i, h in enumerate(stage["absolute"]):
                values[h] = out[i]
        for h in stage["others"]:
            q_coo = self.heads[h]["q"]
            q = self._project(ctx, q_coo, max(int(q_coo[1].max()) + 1 if len(q_coo[1]) else 1, 2))
            kind = self.head_kind[h]
            if kind == "absolute":
                address = q[0]
                ctx["errors"][0] += ((address < 0) | (address > ctx["positions"])).sum().to(torch.int32)
                winner = address.clamp(min=0)
            elif kind == "match":
                winner = self._match(h, ctx, q, state)
            elif kind == "latest":
                winner = self._latest(h, q, ctx["positions"], tokens, state)
            else:
                winner = self._scan(h, q, ctx["positions"], tokens)
            values[h] = self._project_inputs(self.heads[h]["v"], 1, tokens, winner)[0]
        rows, results = [], []
        for h in stage["heads"]:  # Packed heads read the same residual.
            for _column, val, row in zip(*self.heads[h]["o"]):
                rows.append(int(row))
                results.append(values[h] * int(val))
        self._publish(ctx, rows, results)

    # -- FFN ---------------------------------------------------------------------

    def _plan(self, stage, inv):
        """Unit ids to run once per group and at every position, given which
        physical rows are in group form."""
        key = inv[stage["reads"]].tobytes()
        plan = stage["plans"].get(key)
        if plan is None:
            counts = np.concatenate([[0], np.cumsum(~inv[stage["reads"]])])
            starts = stage["starts"]
            shared = counts[starts[1:]] == counts[starts[:-1]]
            plan = (
                torch.tensor(stage["ids"][shared], dtype=torch.int32, device=self.device),
                torch.tensor(stage["ids"][~shared], dtype=torch.int32, device=self.device),
                shared,
                torch.as_tensor(shared, device=self.device, dtype=torch.int8),
            )
            stage["plans"][key] = plan
        return plan

    def _ffn(self, stage, ctx):
        shared_ids, own_ids, shared, shared_device = self._plan(stage, ctx["inv"])
        u = self.units
        for ids, mode, count in ((shared_ids, 0, ctx["groups"]), (own_ids, 1, ctx["n"])):
            if len(ids) and count:
                _ffn_kernel[(len(ids), triton.cdiv(count, self.block))](
                    ctx["F"],
                    ctx["L"],
                    ctx["F"].stride(0),
                    ctx["L"].stride(0),
                    ctx["inv_device"],
                    ctx["group"],
                    count,
                    ids,
                    u["slot"],
                    u["neurons"],
                    u["entries"],
                    u["rows"],
                    u["vals"],
                    u["out"],
                    u["t_row"],
                    u["t_low"],
                    u["t_start"],
                    u["t_span"],
                    self.tables,
                    BLOCK=self.block,
                    MODE=mode,
                )
        if not (ctx["share"] and len(own_ids)):
            ctx["inv"][stage["slots"]] = shared
            ctx["inv_device"][stage["slots_device"]] = shared_device
            return
        # Units with varying inputs can still be constant per group (position - phase).
        rows = self.units["slot"][own_ids.long()]
        flags = torch.zeros(len(rows), dtype=torch.int32, device=self.device)
        _differs_kernel[(len(rows), triton.cdiv(ctx["n"], self.block))](
            ctx["F"], ctx["F"].stride(0), rows, ctx["group"], ctx["leaders"], ctx["n"], flags, BLOCK=self.block
        )
        entry = self._decide(ctx, flags != 0, id(own_ids))
        if "converged" not in entry:
            entry["converged"] = rows[entry["device"]]
            after = shared.copy()
            after[~shared] = entry["same"]
            entry["after"] = after
            entry["after_device"] = torch.as_tensor(after, device=self.device, dtype=torch.int8)
        if len(entry["converged"]):
            _leaders_kernel[(len(entry["converged"]), triton.cdiv(ctx["groups"], self.block))](
                ctx["F"],
                ctx["L"],
                ctx["F"].stride(0),
                ctx["L"].stride(0),
                entry["converged"],
                ctx["leaders"],
                ctx["groups"],
                BLOCK=self.block,
            )
        ctx["inv"][stage["slots"]] = entry["after"]
        ctx["inv_device"][stage["slots_device"]] = entry["after_device"]

    # -- Forward -------------------------------------------------------------

    def _layout(self, host_positions):
        """Each position's group and each group's first position (cached for consecutive runs)."""
        n = len(host_positions)
        if self.period is None:
            return np.arange(n), np.arange(n), False
        period, offset = self.period
        consecutive = n and host_positions[-1] - host_positions[0] == n - 1
        key = (n, int(host_positions[0] + offset) % period) if consecutive else None
        cache = self.__dict__.setdefault("_layouts", {})
        if key in cache:
            return cache[key]
        gid = (host_positions + offset) // period
        starts = np.ones(n, dtype=bool)
        starts[1:] = (gid[1:] != gid[:-1]) | (host_positions[1:] != host_positions[:-1] + 1)
        layout = np.cumsum(starts) - 1, np.flatnonzero(starts), True
        if key is not None:
            cache[key] = layout
        return layout

    def _forward(self, ctx, tokens, state):
        """One chunk at ctx's positions and groups. With ctx["replay"] set it never
        waits for the device, so it can be captured as a CUDA graph."""
        positions = ctx["positions"]
        ctx["n"], ctx["step"] = len(positions), 0
        for name in ("violations", "refinements", "outside", "errors"):
            ctx[name].zero_()
        ctx["inv"][:] = False
        ctx["inv"][self.zero_row] = True
        columns = list(range(self.input_width))
        self._publish(
            ctx, columns, [torch.ones_like(positions) if c == 0 else self._input(c, tokens, positions) for c in columns]
        )
        for stage in self.stages:
            if stage["kind"] == "attention":
                self._attention(stage, ctx, tokens, state)
            else:
                self._ffn(stage, ctx)
        if self.output_table is not None:
            row, low, table = self.output_table
            index = self._row(ctx, row) - low
            inside = (index >= 0) & (index < len(table))
            ctx["outside"] += (~inside).sum()
            return table[torch.where(inside, index, 0)], inside
        return self._logits(ctx).argmax(dim=1) + self.lower, None

    def _logits(self, ctx):
        logits = torch.zeros((ctx["n"], self.vocab), dtype=torch.int64, device=self.device)
        for i, row in enumerate(self.output_phys.tolist()):
            logits += self._row(ctx, row)[:, None] * self.unembedding[i][None, :]
        return logits

    def positions(self, length, start):
        """Positions to evaluate: those predicting tokens start.., plus earlier tagged prompt positions."""
        needed = set()
        for meta in self.match.values():
            first = meta["start"] + (meta["phase"] - meta["start"]) % meta["period"]
            stop = min(start - 1, meta.get("end", start - 1))
            needed.update(range(first, max(first, stop), meta["period"]))
        tagged = np.array(sorted(p for p in needed if p < start - 1), dtype=np.int64)
        return np.concatenate([tagged, np.arange(start - 1, length, dtype=np.int64)])

    @torch.no_grad()
    def predict(self, tokens, start):
        """The model's next token after each of positions start-1 .. len(tokens)-1.

        Entry i predicts tokens[start + i]; the last entry predicts the token after the trace.
        """
        if not 1 <= start <= len(tokens) <= self.max_context:
            raise ValueError("start or length outside the compiled context")
        if len(tokens) >= 1 << (POSITION_BITS - 1):
            raise ValueError("trace too long for packed positions")
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(np.asarray(tokens, dtype=np.int64))
        tokens = tokens.to(self.device).long()  # Copy narrow tokens, then widen.
        all_positions = self.positions(len(tokens), start)
        size = min(self.chunk, len(all_positions))
        layouts = [self._layout(all_positions[a : a + self.chunk]) for a in range(0, len(all_positions), self.chunk)]
        groups = max(len(leaders) for _, leaders, _ in layouts)
        ctx = {
            "F": torch.zeros((self.rows, size), dtype=torch.int64, device=self.device),
            "L": torch.zeros((self.rows, groups), dtype=torch.int64, device=self.device),
            "inv": np.zeros(self.rows, dtype=bool),
            "inv_device": torch.zeros(self.rows, dtype=torch.int8, device=self.device),
            "errors": torch.zeros(len(ERRORS), dtype=torch.int32, device=self.device),
        }
        for name in ("violations", "refinements", "outside"):
            ctx[name] = torch.zeros((), dtype=torch.int64, device=self.device)
        state, outputs, plan, hint, graphs, replayed, serial = {}, [], None, None, {}, set(), 0
        self.stats = dict(chunks=0, discovered=0, redone=0, graphed=0)  # Of this call, for tuning.
        for a in range(0, len(all_positions), self.chunk):
            chunk = all_positions[a : a + self.chunk]
            group, leaders, share = layouts[a // self.chunk]
            while True:
                # Replay the last chunk's decisions with no syncs (as a CUDA graph
                # once a plan has run eagerly); if one turns out wrong, discover anew.
                key = (plan["serial"] if plan else None, len(chunk), len(leaders))
                if plan is not None and self.graphs and plan["serial"] in replayed:
                    if key not in graphs:
                        graphs[key] = self._capture(ctx, plan, chunk, group, leaders, share, tokens, state)
                    graph, static, results = graphs[key]
                    for buffer, host in zip(static, (chunk, group, leaders)):
                        buffer.copy_(torch.as_tensor(host))
                    graph.replay()
                    predicted, inside, commit = results
                    predicted = predicted.clone()
                else:
                    ctx.update(
                        positions=torch.as_tensor(chunk, device=self.device),
                        group=torch.as_tensor(group, device=self.device),
                        leaders=torch.as_tensor(leaders, device=self.device),
                        groups=len(leaders),
                        share=share,
                        replay=plan and plan["entries"],
                        record=[],
                        commit=[],
                        hint=hint,
                    )
                    predicted, inside = self._forward(ctx, tokens, state)
                    commit = ctx["commit"]
                checks = torch.stack([ctx["violations"], ctx["refinements"], ctx["outside"]]).tolist()
                self.stats["chunks"] += 1
                self.stats["graphed"] += plan is not None and key in graphs
                self.stats["discovered"] += plan is None
                if plan is not None and checks[0]:
                    self.stats["redone"] += 1
                    plan, hint = None, plan["entries"]
                    continue
                break
            for register in commit:
                register()
            if plan is None:
                plan, serial, hint = {"serial": serial, "entries": ctx["record"]}, serial + 1, None
            else:
                replayed.add(plan["serial"])
            for count, message in zip(ctx["errors"].tolist(), ERRORS):
                if count:
                    raise ValueError(message)
            if inside is not None and checks[2]:
                predicted = torch.where(inside, predicted, self._logits(ctx).argmax(dim=1) + self.lower)
            outputs.append(predicted)
        predicted = torch.cat(outputs)
        return predicted[len(predicted) - (len(tokens) - start + 1) :]

    def _capture(self, ctx, plan, chunk, group, leaders, share, tokens, state):
        """A CUDA graph of one replayed chunk, with static position and group buffers."""
        static = [torch.as_tensor(host, device=self.device) for host in (chunk, group, leaders)]
        ctx.update(
            positions=static[0],
            group=static[1],
            leaders=static[2],
            groups=len(leaders),
            share=share,
            replay=plan["entries"],
            record=[],
            commit=[],
            hint=None,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            predicted, inside = self._forward(ctx, tokens, state)
        return graph, static, (predicted, inside, ctx["commit"])

    def check(self, tokens, start):
        """(first position from `start` where tokens differ from the model's prediction
        or None, the model's token there)."""
        predicted = self.predict(tokens, start)
        actual = tokens[start:] if isinstance(tokens, torch.Tensor) else np.asarray(tokens[start:], dtype=np.int64)
        actual = torch.as_tensor(actual).to(self.device).long()
        wrong = torch.nonzero(predicted[:-1] != actual)
        if len(wrong):
            i = int(wrong[0])
            return start + i, int(predicted[i])
        return None, int(predicted[-1])
