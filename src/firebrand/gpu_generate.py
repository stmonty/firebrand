"""Serial generation on a GPU, one position per step (experimental).

GPUGenerator runs a compiled model's forward pass for one new position at a
time, with the same exact int64 kernels and attention indexes as gpu.Verifier,
and keeps its key indexes on the device between steps. A whole step (inputs,
every stage, the output token, index updates) is captured once as a CUDA graph
and replayed, so per-step time is GPU time, not Python overhead. It exists to
measure how step time depends on a model's depth and width.
"""

import torch

from .gpu import TABLE_KEY_BITS, Verifier, _absolute_kernel, _ffn_kernel

BLOCK = 16


class GPUGenerator:
    def __init__(self, model, device="cuda"):
        v = self.v = Verifier(model, device=device, share=False, graphs=False)
        self.model, self.device = model, v.device
        self.vocab = v.vocab
        unsupported = [
            h
            for stage in v.stages
            if stage["kind"] == "attention"
            for h in stage["others"]
            if v.head_kind[h] not in ("match", "latest")
            or (v.head_kind[h] == "match" and 8 * v.match[h]["digits"] > TABLE_KEY_BITS)
        ]
        if unsupported:
            raise ValueError(f"heads {unsupported} need a scanning index; not supported here")
        self.ffn = {
            id(stage): torch.as_tensor(stage["ids"], device=self.device) for stage in v.stages if stage["kind"] == "ffn"
        }
        self.coos = {}
        self.output_rows = torch.as_tensor(v.output_phys, device=self.device)

    # -- State --------------------------------------------------------------

    def _reset(self, capacity):
        v, dev = self.v, self.device
        self.x = torch.zeros((v.rows, 1), dtype=torch.int64, device=dev)
        self.unused = torch.zeros((v.rows, 1), dtype=torch.int64, device=dev)
        self.inv = torch.zeros(v.rows, dtype=torch.int8, device=dev)
        self.group = torch.zeros(1, dtype=torch.int64, device=dev)
        self.position = torch.zeros(1, dtype=torch.int64, device=dev)
        self.tokens = torch.zeros(capacity + 1, dtype=torch.int64, device=dev)
        self.errors = torch.zeros(3, dtype=torch.int32, device=dev)
        self.tables, self.sentinels, self.newest = {}, {}, {}
        for stage in v.stages:
            if stage["kind"] != "attention":
                continue
            for h in stage["others"]:
                if v.head_kind[h] == "match":
                    self.tables[h] = torch.full(((1 << TABLE_KEY_BITS) + 1,), -1, dtype=torch.int64, device=dev)
                    self.sentinels[h] = torch.full((1,), -1, dtype=torch.int64, device=dev)
                else:
                    self.newest[h] = torch.full((self.vocab,), -1, dtype=torch.int64, device=dev)
        self.outs = {
            id(stage): torch.zeros((len(stage["absolute"]), 1), dtype=torch.int64, device=dev)
            for stage in v.stages
            if stage["kind"] == "attention" and stage["absolute"]
        }

    def _device(self, coo):
        """A projection's (rows, cols, vals) as device tensors, made once (graphs cannot copy from the host)."""
        key = id(coo)
        if key not in self.coos:
            self.coos[key] = tuple(torch.as_tensor(a, device=self.device) for a in coo)
        return self.coos[key]

    def _project(self, coo, dims):
        rows, cols, vals = self._device(coo)
        out = torch.zeros(dims, dtype=torch.int64, device=self.device)
        if len(rows):
            out.index_add_(0, cols, self.x[rows, 0] * vals)
        return out

    def _register_latest(self, positions):
        """Index the tokens at `positions` for every latest head (their keys read only inputs)."""
        for h, newest in self.newest.items():
            k = self.v._project_inputs(self.v.heads[h]["k"], 4, self.tokens, positions)
            tagged = k[3] == 1
            value = (k[0] - self.v.lower).clamp(0, self.vocab - 1)
            newest.scatter_reduce_(0, torch.where(tagged, value, 0), torch.where(tagged, positions, -1), "amax")

    # -- One forward pass ------------------------------------------------------

    def _forward(self):
        """The next token after position self.position, from the indexes and residual."""
        v, p = self.v, self.position
        self.x[0].fill_(1)
        for column in range(1, v.input_width):
            self.x[column].copy_(v._input(column, self.tokens, p))
        for stage in v.stages:
            if stage["kind"] == "attention":
                self._attention(stage, p)
            else:
                u = v.units
                ids = self.ffn[id(stage)]
                _ffn_kernel[(len(ids), 1)](
                    self.x,
                    self.unused,
                    1,
                    1,
                    self.inv,
                    self.group,
                    1,
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
                    v.tables,
                    BLOCK=BLOCK,
                    MODE=1,
                )
        if v.output_table is not None:
            row, low, table = v.output_table
            return table[(self.x[row] - low).clamp(0, len(table) - 1)][0]  # A 1-element index: no sync.
        logits = (self.x[self.output_rows, 0][:, None] * v.unembedding).sum(0)
        return logits.argmax() + v.lower

    def _attention(self, stage, p):
        v, values = self.v, {}
        if stage["absolute"]:
            out = self.outs[id(stage)]
            _absolute_kernel[(len(stage["absolute"]), 1)](
                self.x,
                self.unused,
                1,
                1,
                self.inv,
                self.group,
                1,
                p,
                self.tokens,
                v.embedding,
                v.embedding_columns,
                v.lower,
                out,
                out.stride(0),
                self.errors,
                *stage["arrays"],
                BLOCK=BLOCK,
            )
            for i, h in enumerate(stage["absolute"]):
                values[h] = out[i]
        for h in stage["others"]:
            q_coo = v.heads[h]["q"]
            q = self._project(q_coo, max(int(q_coo[1].max()) + 1 if len(q_coo[1]) else 1, 2))
            if v.head_kind[h] == "match":
                winner = self._match(h, q, p)
            else:
                winner = self._latest(h, q)
            values[h] = v._project_inputs(v.heads[h]["v"], 1, self.tokens, winner)[0]
        for h in stage["heads"]:  # Packed heads read the same residual.
            _columns, vals, rows = v.heads[h]["o"]
            for val, row in zip(vals.tolist(), rows.tolist()):
                self.x[row] = values[h] * val

    def _match(self, h, q, p):
        meta, table, sentinel = self.v.match[h], self.tables[h], self.sentinels[h]
        digits = meta["digits"]
        k = self._project(self.v.heads[h]["k"], 2 * digits + 3)
        tagged, is_sentinel = k[2 * digits + 2] == 1, k[2 * digits] == 1
        key, key_ok = self.v._pack(k[:digits, None].double())
        query, query_ok = self.v._pack(q[:digits, None].double() / (2 * meta["scale"]))
        dummy = len(table) - 1
        table.scatter_reduce_(0, torch.where(tagged & key_ok, key, dummy), torch.where(tagged & key_ok, p, -1), "amax")
        sentinel.copy_(torch.where(is_sentinel, p, sentinel))
        found = table[torch.where(query_ok, query, dummy)]
        self.errors[1] += ((found < 0) & (sentinel < 0)).sum().to(torch.int32)
        return torch.where(found >= 0, found, sentinel.clamp(min=0))

    def _latest(self, h, q):
        """Nearest tagged value to the query, then the newest such position."""
        newest = self.newest[h]
        key = torch.div(q[0], -2 * q[1], rounding_mode="floor") - self.v.lower
        values = torch.arange(self.vocab, device=self.device)
        distance = torch.where(newest >= 0, (values - key) ** 2, torch.iinfo(torch.int64).max)
        best = distance.min()
        winner = torch.where(distance == best, newest, -1).max()
        self.errors[2] += (winner < 0).to(torch.int32)
        return winner.reshape(1)

    def _step(self):
        token = self._forward()
        self.tokens.index_copy_(0, self.position + 1, token.reshape(1))
        self.position += 1
        self._register_latest(self.position)

    # -- Generation -------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prefix, max_steps, *, check_every=256):
        """model.generate(prefix, max_steps) on the GPU, one CUDA graph replay per token."""
        halts = set(self.model.config["halt_values"])
        capacity = min(self.model.config["max_context"], len(prefix) + max_steps)
        self._reset(capacity)
        self.tokens[: len(prefix)] = torch.as_tensor(prefix, device=self.device)
        self._register_latest(torch.arange(len(prefix), device=self.device))
        for p in self.v.positions(len(prefix), len(prefix))[:-1]:  # Match keys at tagged prompt positions.
            self.position.fill_(int(p))
            self._forward()
        self.position.fill_(len(prefix) - 1)
        if prefix[-1] in halts or not max_steps:
            return list(prefix)
        self._step()  # Eager once, then capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        length, steps = len(prefix) + 1, 1
        while steps < max_steps and length <= capacity - 1:
            batch = min(check_every, max_steps - steps, capacity - length)
            for _ in range(batch):
                graph.replay()
            new = self.tokens[length - 1 : length + batch].tolist()
            for i, token in enumerate(new):
                if token in halts:
                    return self.tokens[: length + i].tolist()
            length, steps = length + batch, steps + batch
        out = self.tokens[:length].tolist()
        for i in range(len(prefix), len(out)):
            if out[i] in halts:
                return out[: i + 1]
        return out

    def step_time(self, prefix, steps=2000):
        """Seconds per generated token, timing graph replays only."""
        steps = min(steps, self.model.config["max_context"] - len(prefix) - 4)
        self.generate(prefix, 2)
        self._reset(min(self.model.config["max_context"], len(prefix) + steps + 4))
        self.tokens[: len(prefix)] = torch.as_tensor(prefix, device=self.device)
        self._register_latest(torch.arange(len(prefix), device=self.device))
        for p in self.v.positions(len(prefix), len(prefix))[:-1]:
            self.position.fill_(int(p))
            self._forward()
        self.position.fill_(len(prefix) - 1)
        self._step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / 1000 / steps
