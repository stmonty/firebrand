"""Export a compiled Transformer as a plain PyTorch module.

A decoder-only transformer with no normalization: token embedding plus fixed
positional features, residual attention and ReLU FFN blocks, an unembedding.
Every position is computed in parallel with dense causal attention, in float64.
Attention is hard by default; attention="softmax" uses softmax(beta * scores),
which gives the same outputs because the winning score leads by at least 1 and
exp(-beta) underflows to zero. Dense attention is O(length**2), so this is for
short contexts; gpu.py handles long ones.
"""

import numpy as np
import torch
from torch import nn

from .runtime import Sparse


def _tensor(weight, rows=None):
    """A float64 tensor of a weight, padded or cut to `rows` rows; sparse
    weights become sparse COO tensors."""
    if isinstance(weight, Sparse):
        shape = (weight.shape[0] if rows is None else rows, weight.shape[1])
        keep = weight.rows < shape[0]
        indices = torch.tensor(np.stack([weight.rows[keep], weight.cols[keep]]), dtype=torch.int64)
        values = torch.tensor(weight.vals[keep], dtype=torch.float64)
        return torch.sparse_coo_tensor(indices, values, shape, check_invariants=True).coalesce()
    weight = np.asarray(weight, dtype=np.float64)
    if rows is not None:
        padded = np.zeros((rows, weight.shape[1]))
        n = min(rows, weight.shape[0])
        padded[:n] = weight[:n]
        weight = padded
    return torch.tensor(weight)


def _matmul(x, weight):
    if weight.is_sparse:
        return torch.sparse.mm(weight.t(), x.t()).t()
    return x @ weight


class TorchTransformer(nn.Module):
    def __init__(self, model, *, attention="hardmax", beta=1000.0):
        super().__init__()
        if attention not in ("hardmax", "softmax"):
            raise ValueError("attention must be 'hardmax' or 'softmax'")
        c, w = model.config, model.weights
        self.config, self.attention, self.beta = c, attention, float(beta)
        self.width, self.lower = c["width"], c["lower"]
        self.register_buffer("embedding", _tensor(w["embedding"]))
        self.register_buffer("unembedding", _tensor(w["unembedding"]))
        schedule = c.get("schedule") or (
            [{"kind": "attention", "index": h} for h in range(c["heads"])]
            + [{"kind": "ffn", "index": i} for i in range(c["layers"])]
        )
        # Blocks in schedule order: ("attention", [head, ...]) or ("ffn", index).
        self.blocks = []
        for step in schedule:
            if step["kind"] == "attention":
                heads = step.get("heads", [step.get("index")])
                for h in heads:
                    # Keys and values may read only the input columns; as residual
                    # projections they get zero rows for every other column.
                    for part in "qkvo":
                        rows = None if part in "qo" else self.width
                        self.register_buffer(f"head_{h}_{part}", _tensor(w[f"head_{h}_{part}"], rows))
                self.blocks.append(("attention", list(heads)))
            elif step["kind"] == "ffn":
                i = step["index"]
                self.register_buffer(f"ff_{i}_in", _tensor(w[f"ff_{i}_in"]))
                self.register_buffer(f"ff_{i}_out", _tensor(w[f"ff_{i}_out"]))
                self.blocks.append(("ffn", i))
            else:
                raise ValueError(f"unsupported schedule step: {step['kind']}")

    def embed(self, tokens):
        """Token embedding plus the fixed positional coordinates: position,
        position squared, and the configured features (deterministic functions
        of the index, like sinusoidal encodings)."""
        c = self.config
        n = len(tokens)
        rows = self.embedding[tokens - self.lower]
        x = torch.zeros((n, self.width), dtype=torch.float64, device=tokens.device)
        columns = min(self.width, rows.shape[1])
        x[:, :columns] = rows[:, :columns]
        p = torch.arange(n, device=tokens.device, dtype=torch.int64)
        x[:, 1], x[:, 2] = p.double(), (p * p).double()
        for f in c.get("position_features", ()):
            kind, column = f["kind"], f["column"]
            if kind == "mod":
                value = (p + f["offset"]) % f["period"]
            elif kind == "tag":
                end = f.get("end")
                value = (p >= f["start"]) & (p % f["period"] == f["phase"])
                if end is not None:
                    value &= p < end
            elif kind == "digit":
                value = (torch.clamp(p + f["offset"], min=0) >> (8 * f["power"])) & 255
            elif kind == "quotient":
                shifted = p + f["offset"]
                value = torch.where(
                    shifted >= 0, torch.div(shifted, f["period"], rounding_mode="floor"), torch.full_like(shifted, -1)
                )
            elif kind == "point":
                value = p == f["position"]
            elif kind == "token_squared":
                continue  # Stored in the embedding.
            else:
                raise ValueError(f"unsupported position feature: {kind}")
            x[:, column] = value.double()
        return x

    def _attend(self, x, h):
        q = _matmul(x, getattr(self, f"head_{h}_q"))
        k = _matmul(x, getattr(self, f"head_{h}_k"))
        v = _matmul(x, getattr(self, f"head_{h}_v"))
        scores = q @ k.t()
        n = len(x)
        future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(future, -torch.inf)
        if self.attention == "hardmax":
            # torch.argmax returns the first maximum, like the NumPy decoder.
            mixed = v[scores.argmax(dim=1)]
        else:
            mixed = torch.softmax(self.beta * (scores - scores.amax(dim=1, keepdim=True)), dim=1) @ v
        return _matmul(mixed, getattr(self, f"head_{h}_o"))

    def forward(self, tokens):
        """Logits for the next token at every position of a 1-D token tensor."""
        x = self.embed(tokens)
        for kind, index in self.blocks:
            if kind == "attention":
                # A packed block's heads all read the same residual.
                x = x + sum(self._attend(x, h) for h in index)
            else:
                hidden = torch.relu(_matmul(x, getattr(self, f"ff_{index}_in")))
                x = x + _matmul(hidden, getattr(self, f"ff_{index}_out"))
        return _matmul(x, self.unembedding)

    @torch.no_grad()
    def generate(self, prefix, max_steps=100):
        """Greedy decoding, recomputing the whole sequence each step (slow, but self-contained)."""
        device = self.embedding.device
        tokens = list(prefix)
        for _ in range(max_steps):
            if tokens[-1] in self.config["halt_values"]:
                break
            logits = self(torch.tensor(tokens, device=device))
            tokens.append(int(logits[-1].argmax()) + self.lower)
        return tokens


def to_torch(model, **options):
    """A TorchTransformer computing the same function as `model`."""
    return TorchTransformer(model, **options)
