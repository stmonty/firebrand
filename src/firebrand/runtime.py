"""Matrix-only decoder. This module does not import or execute the source IR."""

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sparse:
    """A coordinate-list matrix (duplicates are summed); supports `vector @ Sparse`."""

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    shape: tuple

    __array_ufunc__ = None  # Make `vector @ Sparse` defer to __rmatmul__.

    def __rmatmul__(self, x):
        if x.ndim != 1:
            raise ValueError("sparse weights multiply single residual vectors")
        return np.bincount(self.cols, weights=self.vals * x[self.rows], minlength=self.shape[1])

    def toarray(self):
        dense = np.zeros(self.shape)
        np.add.at(dense, (self.rows, self.cols), self.vals)
        return dense


@dataclass
class Transformer:
    config: dict
    weights: dict[str, np.ndarray]

    def _validate_prefix(self, tokens):
        c = self.config
        if len(tokens) < c["required_prefix"] or len(tokens) > c["max_context"]:
            raise ValueError("prefix length outside compiled context contract")
        if any(type(t) is not int or not c["lower"] <= t <= c["upper"] for t in tokens):
            raise ValueError("token outside compiled alphabet")

    def logits(self, tokens, *, probes=False):
        """Next-token logits by the reference decoder: every head scores every eligible position.

        Match-head keys come from earlier forward passes at each tagged position.
        """
        self._validate_prefix(tokens)
        x = self._embed(tokens, width=self._input_width)
        stored = {int(h): [] for h in self.config.get("match_heads", {})}

        def attend(index, q, kind, residual, length):
            if kind == "match":
                return self._scan_match(stored[index], index, q, residual, length, x)
            k, v = self._project(x[:length], index)
            return _scan(k, v, q, kind == "latest")

        for p in self._match_positions(len(tokens) - 1):
            self._forward(self._embed(tokens[p : p + 1], p)[0], p + 1, attend)
        return self._forward(self._embed(tokens[-1:], len(tokens) - 1)[0], len(tokens), attend, probes=probes)

    def _match_positions(self, end):
        """Positions before end tagged for some match head, in order."""
        positions = set()
        for head in self.config.get("match_heads", {}).values():
            first = head["start"] + (head["phase"] - head["start"]) % head["period"]
            positions.update(range(first, min(end, head.get("end", end)), head["period"]))
        return sorted(positions)

    def _match_key(self, index, residual):
        """(digits, tagged, sentinel) of the current position's key for a match head."""
        head = self.config["match_heads"][str(index)]
        m = head["digits"]
        k = residual @ self.weights[f"head_{index}_k"]
        return tuple(int(d) for d in k[:m]), k[2 * m + 2] == 1, k[2 * m] == 1

    def _query_digits(self, index, q):
        head = self.config["match_heads"][str(index)]
        return tuple(int(d) for d in q[: head["digits"]] / (2 * head["scale"]))

    def _scan_match(self, stored, index, q, residual, length, x):
        _digits, tagged, _ = self._match_key(index, residual)
        k = residual @ self.weights[f"head_{index}_k"]
        if tagged or k[2 * self.config["match_heads"][str(index)]["digits"]] == 1:
            stored.append((length - 1, k))
        if not stored:
            raise ValueError("no sentinel in causal prefix")
        scores = [q @ key for _, key in stored]
        winner, key = stored[int(np.argmax(scores))]
        m = self.config["match_heads"][str(index)]["digits"]
        if tuple(int(d) for d in key[:m]) != self._query_digits(index, q) and key[2 * m] != 1:
            raise ValueError("no sentinel in causal prefix")
        return self._values(x[winner : winner + 1], index)[0]

    @property
    def _input_width(self):
        """Columns that keys and values can read: token and position features."""
        return self.config.get("input_width", self.config["width"])

    def _embed(self, tokens, start=0, width=None):
        c, w = self.config, self.weights
        width = c["width"] if width is None else width
        rows = w["embedding"][np.asarray(tokens) - c["lower"]]
        x = np.zeros((len(tokens), width))
        columns = min(width, rows.shape[1])
        x[:, :columns] = rows[:, :columns]
        positions = np.arange(start, start + len(tokens), dtype=np.float64)
        x[:, 1], x[:, 2] = positions, positions**2
        # Version 3 positional features: deterministic functions of the index.
        for feature in c.get("position_features", ()):
            if feature["kind"] == "mod":
                x[:, feature["column"]] = (positions + feature["offset"]) % feature["period"]
            elif feature["kind"] == "tag":
                x[:, feature["column"]] = (
                    (positions >= feature["start"])
                    & (positions < feature.get("end", np.inf))
                    & (positions % feature["period"] == feature["phase"])
                )
            elif feature["kind"] in ("digit", "quotient"):
                shifted = positions + feature["offset"]
                if feature["kind"] == "digit":
                    value = (np.maximum(shifted, 0).astype(np.int64) >> (8 * feature["power"])) & 255
                else:
                    value = np.where(shifted >= 0, np.floor_divide(shifted, feature["period"]), -1)
                x[:, feature["column"]] = value
            elif feature["kind"] == "point":
                x[:, feature["column"]] = positions == feature["position"]
            elif feature["kind"] != "token_squared":  # Stored in the embedding.
                raise ValueError(f"unsupported position feature: {feature['kind']}")
        return x

    def _values(self, x, index):
        """Value projections alone (match heads compute keys from hidden state)."""
        v = self.weights[f"head_{index}_v"]
        n = min(x.shape[1], v.shape[0])
        return x[:, :n] @ v[:n]

    def _project(self, x, index):
        # Rows beyond the input columns are zero: keys read only immutable inputs.
        k, v = self.weights[f"head_{index}_k"], self.weights[f"head_{index}_v"]
        n = min(x.shape[1], k.shape[0])
        return x[:, :n] @ k[:n], x[:, :n] @ v[:n]

    def _forward(self, residual, length, attend, *, probes=False):
        """Run the schedule on the final position. attend(index, q, kind, residual,
        length) returns the value row of the head's hardmax winner."""
        c, w = self.config, self.weights
        snapshots = {}
        schedule = c.get("schedule")
        if schedule is None:
            # Version 1 checkpoints placed all reads before all FFN blocks.
            schedule = [{"kind": "attention", "index": h} for h in range(c["heads"])] + [
                {"kind": "ffn", "index": i} for i in range(c["layers"])
            ]
        kinds = c.get("head_kinds", ())
        for step in schedule:
            if step["kind"] == "attention":
                # Packed stages hold several heads that all read the same residual.
                heads = step.get("heads", [step.get("index")])
                update = 0
                for index in heads:
                    # Queries may depend on previous blocks in this forward pass.
                    # K/V only read the immutable positional/token coordinates, so
                    # their projections equal those from full per-layer residuals.
                    q = residual @ w[f"head_{index}_q"]
                    kind = kinds[index] if index < len(kinds) else "absolute"
                    if kind == "absolute" and c.get("addressing") == "absolute_checked":
                        address = q[0]
                        if not np.isfinite(address) or address != np.floor(address) or not 0 <= address < length:
                            raise ValueError("address outside causal prefix")
                    update = update + attend(index, q, kind, residual, length) @ w[f"head_{index}_o"]
                residual = residual + update
                stage = step.get("name", f"head_{heads[0]}")
                if probes and c["format_version"] == 1:
                    snapshots["attention"] = residual.copy()
            elif step["kind"] == "ffn":
                index = step["index"]
                residual = residual + np.maximum(0, residual @ w[f"ff_{index}_in"]) @ w[f"ff_{index}_out"]
                stage = step.get("name", f"ff_{index}")
            else:
                raise ValueError(f"unsupported schedule step: {step['kind']}")
            if probes:
                snapshots[stage] = residual.copy()
        logits = residual @ w["unembedding"]
        return (logits, snapshots) if probes else logits

    def generate(self, prefix, max_steps=100, *, use_cache=True, backend="python"):
        """Greedy generation until a halt token or max_steps.

        backend="python" uses a key/value cache with exact indexes (or the scanning
        reference with use_cache=False); backend="c" uses the compiled decoder.
        """
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        tokens = list(prefix)
        # Validate even when the prefix already halts or the budget is zero.
        self._validate_prefix(tokens)
        if backend == "c":
            from . import fast

            return fast.generate(self, tokens, max_steps)
        if backend != "python":
            raise ValueError("backend must be 'python' or 'c'")
        cache = None
        for _ in range(max_steps):
            if tokens[-1] in self.config["halt_values"]:
                break
            if use_cache:
                if cache is None:
                    capacity = min(self.config["max_context"], len(tokens) + max_steps - 1)
                    cache = _KVCache(self, tokens, capacity)
                else:
                    cache.append(tokens[-1])
                logits = cache.logits()
            else:
                logits = self.logits(tokens)
            tokens.append(int(np.argmax(logits)) + self.config["lower"])
        return tokens

    def save(self, path):
        """Save portable arrays and metadata, with no pickle or source program."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays, sparse = {}, {}
        for name, weight in self.weights.items():
            if isinstance(weight, Sparse):
                sparse[name] = list(weight.shape)
                for part in ("rows", "cols", "vals"):
                    arrays[f"{name}@{part}"] = getattr(weight, part)
            else:
                arrays[name] = weight
        config = dict(self.config, sparse=sparse) if sparse else self.config
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        np.savez(path / "weights.npz", **arrays)

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if config.get("format_version") not in (1, 2, 3, 4):
            raise ValueError("unsupported checkpoint format")
        sparse = config.pop("sparse", {})
        with np.load(path / "weights.npz", allow_pickle=False) as archive:
            weights = {name: archive[name].copy() for name in archive.files if "@" not in name}
            for name, shape in sparse.items():
                weights[name] = Sparse(
                    *(archive[f"{name}@{part}"].copy() for part in ("rows", "cols", "vals")), tuple(shape)
                )
        return cls(config, weights)


def _scan(k, v, q, latest):
    """Reference hardmax: score every key and take the first maximum."""
    winner = int(np.argmax(q @ k.T))
    if latest and k[winner, 3] != 1:
        raise ValueError("no tagged key in causal prefix")
    return v[winner]


class _LatestIndex:
    """Exact winner of a `latest` head without scanning.

    Its keys are (a, a*a, j, tag(j)) and its query (2*M*key, -M, 1, big), so among
    tagged positions the score is M*(key**2 - (a - key)**2) + j with M above every
    j: the nearest value wins, then the newest position. A map from value to its
    newest tagged position, plus the sorted distinct values, answers that
    directly.
    """

    def __init__(self):
        self.newest, self.values = {}, []

    def add(self, value, position):
        if value not in self.newest:
            bisect.insort(self.values, value)
        self.newest[value] = position  # Positions arrive in increasing order.

    def query(self, key):
        if key in self.newest:
            return self.newest[key]
        if not self.values:
            raise ValueError("no tagged key in causal prefix")
        i = bisect.bisect_left(self.values, key)
        candidates = self.values[max(0, i - 1) : i + 1]
        return max(self.newest[v] for v in candidates if (v - key) ** 2 == min((c - key) ** 2 for c in candidates))


class _KVCache:
    """Incremental decoding state. Heads find winners with exact indexes where they
    can (absolute reads, `latest`, `match`) and scan otherwise."""

    def __init__(self, model, prefix, capacity):
        self.model = model
        self.length = len(prefix)
        x = model._embed(prefix, width=model._input_width)
        self.residual = model._embed(prefix[-1:], self.length - 1)[0]
        kinds = model.config.get("head_kinds", ())
        self.buffers, self.indexes, self.matches = {}, {}, {}
        pending = []  # (position, head, value) for latest heads.
        for index in range(model.config["heads"]):
            kind = kinds[index] if index < len(kinds) else "absolute"
            if kind == "match":
                # Keys come from each position's own forward pass (see below).
                v = model._values(x, index)
                k = np.zeros((len(prefix), 1))
                self.matches[index] = ({}, [None])  # digits -> newest position; sentinel.
            else:
                k, v = model._project(x, index)
            keys = np.empty((capacity, *k.shape[1:]), dtype=k.dtype)
            values = np.empty((capacity, *v.shape[1:]), dtype=v.dtype)
            keys[: self.length], values[: self.length] = k, v
            self.buffers[index] = keys, values
            if kind == "latest":
                self.indexes[index] = _LatestIndex()
                pending += [(int(k[j, 2]), index, int(k[j, 0])) for j in np.flatnonzero(k[:, 3] == 1)]
        # Match keys at prompt positions need those positions' forward passes, run
        # in order with every index as it stood at that position.
        pending.sort()
        for p in model._match_positions(self.length - 1):
            while pending and pending[0][0] <= p:
                position, index, value = pending.pop(0)
                self.indexes[index].add(value, position)
            model._forward(model._embed(prefix[p : p + 1], p)[0], p + 1, self._attend)
        for position, index, value in pending:
            self.indexes[index].add(value, position)

    def append(self, token):
        if self.length >= self.model.config["max_context"]:
            raise ValueError("prefix length outside compiled context contract")
        x = self.model._embed([token], self.length)
        for index, (keys, values) in self.buffers.items():
            if index in self.matches:
                values[self.length] = self.model._values(x, index)[0]
                continue
            k, v = self.model._project(x, index)
            keys[self.length], values[self.length] = k[0], v[0]
            if index in self.indexes and k[0, 3] == 1:
                self.indexes[index].add(int(k[0, 0]), int(k[0, 2]))
        self.residual = x[-1]
        self.length += 1

    def _attend(self, index, q, kind, residual, length):
        keys, values = self.buffers[index]
        if kind == "match":
            newest, sentinel = self.matches[index]
            digits, tagged, is_sentinel = self.model._match_key(index, residual)
            if tagged:
                newest[digits] = length - 1
            if is_sentinel:
                sentinel[0] = length - 1
            winner = newest.get(self.model._query_digits(index, q), sentinel[0])
            if winner is None:
                raise ValueError("no sentinel in causal prefix")
            return values[winner]
        if kind == "latest" and index in self.indexes:
            key = q[0] / (-2 * q[1])
            if key == np.floor(key):
                return values[self.indexes[index].query(int(key))]
        elif kind == "absolute" and self.model.config.get("addressing") == "absolute_checked":
            return values[int(q[0])]
        return _scan(keys[:length], values[:length], q, kind == "latest")

    def logits(self, *, probes=False):
        return self.model._forward(self.residual.copy(), self.length, self._attend, probes=probes)
