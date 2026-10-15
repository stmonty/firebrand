"""The compiled decoder (fastrt.c): packing a Transformer for it, building, and running it.

The library is built on first use with the system C compiler, into the user's
cache directory, under a name derived from the source.
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from .runtime import Sparse

SOURCE = Path(__file__).with_name("fastrt.c")
COMMAND = ["cc", "-O2", "-shared", "-fPIC"]
ERRORS = {
    -1: "address outside causal prefix",
    -2: "no tagged key in causal prefix",
    -3: "prefix length outside compiled context contract",
    -4: "out of memory",
    -5: "no sentinel in causal prefix",
}
_library = None


class _Coo(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int32),
        ("row", ctypes.POINTER(ctypes.c_int32)),
        ("col", ctypes.POINTER(ctypes.c_int32)),
        ("val", ctypes.POINTER(ctypes.c_double)),
    ]


_I32, _F64 = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
_COO = ctypes.POINTER(_Coo)


class _Model(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("input_width", ctypes.c_int32),
        ("vocab", ctypes.c_int32),
        ("lower", ctypes.c_int32),
        ("max_context", ctypes.c_int32),
        ("embedding_columns", ctypes.c_int32),
        ("embedding", _F64),
        ("features", ctypes.c_int32),
        ("feature", _I32),
        ("unembedding", _Coo),
        ("heads", ctypes.c_int32),
        ("head_kind", _I32),
        ("head_dims", _I32),
        ("match", _I32),
        ("match_scale", _F64),
        ("q", _COO),
        ("k", _COO),
        ("v", _COO),
        ("o", _COO),
        ("halts", ctypes.c_int32),
        ("halt", _I32),
        ("units", ctypes.c_int32),
        ("unit_head", _I32),
        ("unit_hidden", _I32),
        ("in_start", _I32),
        ("in_row", _I32),
        ("in_neuron", _I32),
        ("in_value", _F64),
        ("out_start", _I32),
        ("out_neuron", _I32),
        ("out_slot", _I32),
        ("out_value", _F64),
        ("write_start", _I32),
        ("writes", _I32),
        ("reader_start", _I32),
        ("readers", _I32),
        ("memo_row", _I32),
        ("memo_low", _I32),
        ("memo_start", _I32),
        ("output_row", ctypes.c_int32),
        ("output_low", ctypes.c_int32),
        ("output_span", ctypes.c_int32),
    ]


def library():
    global _library
    if _library is None:
        source = SOURCE.read_bytes()
        digest = hashlib.sha256(" ".join(COMMAND).encode() + source).hexdigest()[:24]
        path = _cache() / f"fastrt_{digest}.so"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run([*COMMAND, "-o", str(partial), str(SOURCE), "-lm"], check=True)
            partial.replace(path)
        _library = ctypes.CDLL(str(path))
        _library.fastrt_generate.restype = ctypes.c_int64
        _library.fastrt_generate.argtypes = [ctypes.POINTER(_Model), _I32, ctypes.c_int64, ctypes.c_int64]
    return _library


def _cache():
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "firebrand"


class _Packed:
    """Keeps every array referenced by the C structures alive."""

    def __init__(self, model):
        c, w = model.config, model.weights
        self.keep = []
        kinds = c.get("head_kinds", [])
        checked = c.get("addressing") == "absolute_checked"
        codes = {"latest": 1, "match": 3}
        head_kind = [codes.get(kinds[i] if i < len(kinds) else None, 0 if checked else 2) for i in range(c["heads"])]
        matches = c.get("match_heads", {})
        match = [
            [matches[str(i)][f] for f in ("digits", "period", "phase", "start", "sentinel")]
            + [matches[str(i)].get("end", -1)]
            if str(i) in matches
            else [0, 1, 0, 0, 0, -1]
            for i in range(c["heads"])
        ]
        match_scale = [float(matches[str(i)]["scale"]) if str(i) in matches else 0.0 for i in range(c["heads"])]
        heads = [
            (
                self._coo(w[f"head_{i}_q"]),
                self._coo(w[f"head_{i}_k"]),
                self._coo(w[f"head_{i}_v"]),
                self._coo(w[f"head_{i}_o"]),
                w[f"head_{i}_q"].shape[1],
            )
            for i in range(c["heads"])
        ]
        schedule = c.get("schedule") or (
            [{"kind": "attention", "index": h} for h in range(c["heads"])]
            + [{"kind": "ffn", "index": i} for i in range(c["layers"])]
        )
        units = _units(schedule, w, head_kind)
        ranges = _row_ranges(c, len(w["embedding"]))
        memo = _memo(units, ranges)
        output_row, output_low, output_span = _output_memo(self._coo_rows(w["unembedding"]), ranges)
        features = []
        for f in c.get("position_features", []):
            if f["kind"] == "mod":
                features.append((0, f["column"], f["period"], f["offset"], 0, 0))
            elif f["kind"] == "tag":
                features.append((1, f["column"], f["period"], f["phase"], f["start"], f.get("end", -1)))
            elif f["kind"] == "point":
                features.append((3, f["column"], f["position"], 0, 0, 0))
            elif f["kind"] == "digit":
                features.append((4, f["column"], f["power"], f["offset"], 0, 0))
            elif f["kind"] == "quotient":
                features.append((5, f["column"], f["period"], f["offset"], 0, 0))
            else:
                features.append((2, f["column"], 0, 0, 0, 0))
        embedding = np.ascontiguousarray(w["embedding"], dtype=np.float64)
        self.model = _Model(
            # Inputs are the 4 base coordinates plus positional features; small models
            # store full-width embeddings, whose extra columns are zero.
            width=c["width"],
            input_width=4 + len(c.get("position_features", [])),
            vocab=c["upper"] - c["lower"] + 1,
            lower=c["lower"],
            max_context=c["max_context"],
            embedding_columns=embedding.shape[1],
            embedding=self._array(embedding, np.float64, _F64),
            features=len(features),
            feature=self._array(np.array(features or [[0] * 6], dtype=np.int32), np.int32, _I32),
            unembedding=self._coo(w["unembedding"]),
            heads=c["heads"],
            head_kind=self._array(np.array(head_kind or [0], dtype=np.int32), np.int32, _I32),
            head_dims=self._array(np.array([h[4] for h in heads] or [0], dtype=np.int32), np.int32, _I32),
            match=self._array(np.array(match or [[0] * 6], dtype=np.int32), np.int32, _I32),
            match_scale=self._array(np.array(match_scale or [0.0]), np.float64, _F64),
            q=self._structs([h[0] for h in heads]),
            k=self._structs([h[1] for h in heads]),
            v=self._structs([h[2] for h in heads]),
            o=self._structs([h[3] for h in heads]),
            halts=len(c["halt_values"]),
            halt=self._ints(c["halt_values"]),
            units=len(units["head"]),
            unit_head=self._ints(units["head"]),
            unit_hidden=self._ints(units["hidden"]),
            in_start=self._ints(units["in_start"]),
            in_row=self._ints(units["in_row"]),
            in_neuron=self._ints(units["in_neuron"]),
            in_value=self._array(np.array(units["in_value"] or [0.0]), np.float64, _F64),
            out_start=self._ints(units["out_start"]),
            out_neuron=self._ints(units["out_neuron"]),
            out_slot=self._ints(units["out_slot"]),
            out_value=self._array(np.array(units["out_value"] or [0.0]), np.float64, _F64),
            write_start=self._ints(np.cumsum([0] + [len(r) for r in units["writes"]])),
            writes=self._ints(r for rows in units["writes"] for r in rows),
            reader_start=self._ints(np.cumsum([0] + [len(r) for r in _readers(units, c["width"])])),
            readers=self._ints(u for readers in _readers(units, c["width"]) for u in readers),
            memo_row=self._ints(memo["row"]),
            memo_low=self._ints(memo["low"]),
            memo_start=self._ints(memo["start"]),
            output_row=output_row,
            output_low=output_low,
            output_span=output_span,
        )

    def _array(self, array, dtype, pointer):
        array = np.ascontiguousarray(array, dtype=dtype)
        self.keep.append(array)
        return array.ctypes.data_as(pointer)

    def _ints(self, values):
        return self._array(np.array(list(values) or [0], dtype=np.int32), np.int32, _I32)

    def _coo(self, matrix):
        if isinstance(matrix, Sparse):
            rows, cols, vals = matrix.rows, matrix.cols, matrix.vals
        else:
            rows, cols = np.nonzero(matrix)
            vals = matrix[rows, cols]
        return _Coo(
            len(vals),
            self._array(rows, np.int32, _I32),
            self._array(cols, np.int32, _I32),
            self._array(vals, np.float64, _F64),
        )

    @staticmethod
    def _coo_rows(matrix):
        return set(_coo_arrays(matrix)[0].tolist())

    def _structs(self, coos):
        array = (_Coo * max(1, len(coos)))(*coos)
        self.keep.append(array)
        return ctypes.cast(array, _COO)


def _coo_arrays(matrix):
    if isinstance(matrix, Sparse):
        return matrix.rows, matrix.cols, matrix.vals
    rows, cols = np.nonzero(matrix)
    return rows, cols, matrix[rows, cols]


def _units(schedule, w, head_kind):
    """Split the schedule into independent units: each attention head, and each
    FFN node's neurons (a packed FFN stage concatenates nodes, and every node
    writes one slot, so a neuron belongs to the node whose slot it writes)."""
    units = {
        key: []
        for key in (
            "head",
            "hidden",
            "in_row",
            "in_neuron",
            "in_value",
            "out_neuron",
            "out_slot",
            "out_value",
            "reads",
            "writes",
        )
    }
    units["in_start"], units["out_start"] = [0], [0]

    def close():
        units["in_start"].append(len(units["in_row"]))
        units["out_start"].append(len(units["out_neuron"]))

    for step in schedule:
        if step["kind"] == "attention":
            for h in step.get("heads", [step.get("index")]):
                reads = set(_coo_arrays(w[f"head_{h}_q"])[0].tolist())
                if head_kind[h] == 3:  # Match heads compute this position's key.
                    reads |= set(_coo_arrays(w[f"head_{h}_k"])[0].tolist())
                units["head"].append(h)
                units["hidden"].append(0)
                units["reads"].append(sorted(reads))
                units["writes"].append(sorted(set(_coo_arrays(w[f"head_{h}_o"])[1].tolist())))
                close()
            continue
        in_rows, in_cols, in_vals = _coo_arrays(w[f"ff_{step['index']}_in"])
        out_rows, out_cols, out_vals = _coo_arrays(w[f"ff_{step['index']}_out"])
        for slot in np.unique(out_cols):
            chosen = out_cols == slot
            neurons = np.unique(out_rows[chosen])
            local = {int(n): i for i, n in enumerate(neurons)}
            entries = np.isin(in_cols, neurons)
            units["head"].append(-1)
            units["hidden"].append(len(neurons))
            units["in_row"] += in_rows[entries].tolist()
            units["in_neuron"] += [local[int(n)] for n in in_cols[entries]]
            units["in_value"] += in_vals[entries].tolist()
            units["out_neuron"] += [local[int(n)] for n in out_rows[chosen]]
            units["out_slot"] += out_cols[chosen].tolist()
            units["out_value"] += out_vals[chosen].tolist()
            units["reads"].append(sorted(set(in_rows[entries].tolist())))
            units["writes"].append([int(slot)])
            close()
    return units


# A memoized unit keeps a table with one entry per value of its input.
MEMO_SPAN = 1024
MEMO_HIDDEN = 4


def _row_ranges(config, vocab):
    """Integer bounds on residual rows where known: computed slots, the token, and
    bounded positional features (row 0 is the constant one)."""
    ranges = {3: (config["lower"], config["lower"] + vocab - 1)}
    for f in config.get("position_features", []):
        if f["kind"] == "mod":
            ranges[f["column"]] = (0, f["period"] - 1)
        elif f["kind"] in ("tag", "point"):
            ranges[f["column"]] = (0, 1)
        elif f["kind"] == "digit":
            ranges[f["column"]] = (0, 255)
    for entry in config.get("layout", []):
        low, high = entry["range"]
        if int(low) == low and int(high) == high:
            ranges[entry["slot"]] = (int(low), int(high))
    return ranges


def _memo(units, ranges):
    """FFN units whose only input besides the constant row is one row of small
    integer range compute a function of that row's value: the decoder tabulates
    it as values are first seen. Others get row -1 and an empty table."""
    memo = {"row": [], "low": [], "start": [0]}
    for head, hidden, reads in zip(units["head"], units["hidden"], units["reads"]):
        inputs = [r for r in reads if r != 0]
        span = 0
        if head < 0 and hidden >= MEMO_HIDDEN and len(inputs) == 1 and inputs[0] in ranges:
            low, high = ranges[inputs[0]]
            if high - low < MEMO_SPAN:
                span = high - low + 1
        memo["row"].append(inputs[0] if span else -1)
        memo["low"].append(low if span else 0)
        memo["start"].append(memo["start"][-1] + span)
    return memo


def _output_memo(rows, ranges):
    """Like a memoized unit, the output token is tabulated when the unembedding
    reads one row of small range besides the constant: (row, low, span), or
    (-1, 0, 0)."""
    inputs = [r for r in rows if r != 0]
    if len(inputs) == 1 and inputs[0] in ranges:
        low, high = ranges[inputs[0]]
        if high - low < MEMO_SPAN:
            return inputs[0], low, high - low + 1
    return -1, 0, 0


def _readers(units, width):
    readers = [[] for _ in range(width)]
    for unit, rows in enumerate(units["reads"]):
        for row in rows:
            readers[row].append(unit)
    return readers


def generate(model, tokens, max_steps):
    """Extend a validated prefix by up to max_steps tokens with the C decoder."""
    packed = _Packed(model)
    buffer = np.zeros(len(tokens) + max_steps, dtype=np.int32)
    buffer[: len(tokens)] = tokens
    length = library().fastrt_generate(ctypes.byref(packed.model), buffer.ctypes.data_as(_I32), len(tokens), max_steps)
    if length < 0:
        raise ValueError(ERRORS.get(length, f"decoder error {length}"))
    return [int(t) for t in buffer[:length]]


def statistics():
    """(units evaluated, units skipped, memoized results read back) during the
    last compiled-decoder call."""
    lib = library()
    return tuple(
        ctypes.c_int64.in_dll(lib, f"fastrt_{name}").value for name in ("units_run", "units_skipped", "memo_hits")
    )
