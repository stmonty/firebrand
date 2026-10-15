"""Export a compiled model as a stock Hugging Face Phi-3 checkpoint.

The checkpoint loads with AutoModelForCausalLM and generates the compiled
model's tokens exactly, in float64 with SDPA attention (the defaults when
loading it). Prompts start with <bos>, from which layer 0 recovers positions.
"""

import json
from pathlib import Path

import numpy as np

from .runtime import Sparse

HEAD_DIM = 16  # A power of 4: the attention scale 1/sqrt(16) is exact.
ROTARY = 2  # Rotated dimensions per head, left zero.
BETA = 1024  # Softmax inverse temperature.
PIN = 60  # The pinned column holds 2**60.


def _dense(weight):
    return weight.toarray() if isinstance(weight, Sparse) else np.asarray(weight, dtype=np.float64)


def _position_feature(feature, p):
    """A positional input of the compiled model at machine position p (as runtime._embed)."""
    kind = feature["kind"]
    if kind == "mod":
        return (p + feature["offset"]) % feature["period"]
    if kind == "tag":
        end = feature.get("end")
        return int(
            p >= feature["start"] and p % feature["period"] == feature["phase"] and (end is None or end < 0 or p < end)
        )
    if kind == "point":
        return int(p == feature["position"])
    if kind == "digit":
        shifted = p + feature["offset"]
        return (shifted >> (8 * feature["power"])) & 255 if shifted >= 0 else 0
    if kind == "quotient":
        shifted = p + feature["offset"]
        return shifted // feature["period"] if shifted >= 0 else -1
    raise ValueError(f"unsupported position feature: {kind}")


def to_huggingface(model, path, *, max_context=None):
    """Write `model` to `path` as a Phi-3 checkpoint plus tokenizer; returns the path.

    max_context (default: the model's) bounds the exported context, not counting
    <bos>; the first layer's width grows with it.
    """
    import torch
    from transformers import Phi3Config, Phi3ForCausalLM

    c, w = model.config, model.weights
    context = min(max_context or c["max_context"], c["max_context"])
    width, lower, vocab = c["width"], c["lower"], c["upper"] - c["lower"] + 1
    features = [f for f in c.get("position_features", []) if f["kind"] != "token_squared"]
    schedule = c.get("schedule") or (
        [{"kind": "attention", "index": h} for h in range(c["heads"])]
        + [{"kind": "ffn", "index": i} for i in range(c["layers"])]
    )

    # Layers: the position prelude, then each attention stage with the FFN stage after it.
    layers, current = [{"attention": "prelude", "mlp": "prelude"}], None
    for step in schedule:
        if step["kind"] == "attention":
            current = {"attention": step.get("heads", [step.get("index")]), "mlp": None}
            layers.append(current)
        elif current is not None and current["mlp"] is None:
            current["mlp"] = step["index"]
        else:
            current = {"attention": None, "mlp": step["index"]}
            layers.append(current)
    heads = max(1, max((len(layer["attention"]) for layer in layers[1:] if layer["attention"]), default=1))

    # Residual columns: the compiled model's, then the pinned constant, the <bos> flag, and 1/(t+1).
    pin, bos, inverse = width, width + 1, width + 2
    size = 4
    while size < max(width + 3, heads * HEAD_DIM):
        size *= 4
    exponent = PIN - (size.bit_length() - 1) // 2  # The mean of squares is 2**(2*exponent).
    hidden = max(
        [2 * context + 1]
        + [_dense(w[f"ff_{layer['mlp']}_in"]).shape[1] for layer in layers[1:] if layer["mlp"] is not None]
    )
    _certify(model, context, size, exponent)

    tokens = vocab + 2  # <bos> is 0, then the compiled tokens, then <unk>.
    state = {}
    embedding = np.zeros((tokens, size))
    source = _dense(w["embedding"])
    columns = min(source.shape[1], c.get("input_width", width))
    positional = {1, 2} | {f["column"] for f in features}
    for value in range(vocab):
        row = np.zeros(size)
        row[:columns] = source[value, :columns]
        row[sorted(positional)] = 0
        embedding[value + 1] = row
    embedding[0, 0] = 1  # The constant column is 1 everywhere.
    embedding[0, bos] = 1
    embedding[tokens - 1, 0] = 1
    embedding[:, pin] = 2.0**PIN
    state["model.embed_tokens.weight"] = embedding
    gain = np.full(size, 2.0**exponent)
    state["model.norm.weight"] = gain
    lm_head = np.zeros((tokens, size))
    lm_head[1 : vocab + 1, :width] = _dense(w["unembedding"]).T
    lm_head[0, 0] = lm_head[tokens - 1, 0] = -(2.0**40)  # Never generate <bos> or <unk>.
    state["lm_head.weight"] = lm_head

    q_scale = BETA * HEAD_DIM**0.5
    for i, layer in enumerate(layers):
        prefix = f"model.layers.{i}."
        qkv, out = np.zeros((3 * size, size)), np.zeros((size, size))
        gate_up, down = np.zeros((2 * hidden, size)), np.zeros((size, hidden))
        if layer["attention"] == "prelude":
            # Head 0: equal scores (zero query), value = the <bos> flag: the mean is 1/(t+1).
            qkv[2 * size + ROTARY, bos] = 1
            out[inverse, ROTARY] = 1
            _prelude_mlp(gate_up, down, features, context, hidden, inverse)
        else:
            for slot, h in enumerate(layer["attention"] or []):
                base = slot * HEAD_DIM + ROTARY
                for part, offset, scale in (("q", 0, q_scale), ("k", size, 1.0)):
                    matrix = _dense(w[f"head_{h}_{part}"])
                    if matrix.shape[1] > HEAD_DIM - ROTARY:
                        raise ValueError(f"head {h} has more than {HEAD_DIM - ROTARY} query dimensions")
                    qkv[offset + base : offset + base + matrix.shape[1], : matrix.shape[0]] = matrix.T * scale
                v = _dense(w[f"head_{h}_v"])
                qkv[2 * size + base, : v.shape[0]] = v[:, 0]
                out[:width, base] = _dense(w[f"head_{h}_o"])[0]
            if layer["mlp"] is not None:
                w_in, w_out = _dense(w[f"ff_{layer['mlp']}_in"]), _dense(w[f"ff_{layer['mlp']}_out"])
                gate_up[: w_in.shape[1], :width] = w_in.T
                gate_up[hidden : hidden + w_in.shape[1], 0] = 1  # Up = 1: relu(gate) * 1.
                down[:width, : w_out.shape[0]] = w_out.T
        state[prefix + "self_attn.qkv_proj.weight"] = qkv
        state[prefix + "self_attn.o_proj.weight"] = out
        state[prefix + "mlp.gate_up_proj.weight"] = gate_up
        state[prefix + "mlp.down_proj.weight"] = down
        state[prefix + "input_layernorm.weight"] = gain
        state[prefix + "post_attention_layernorm.weight"] = gain

    halts = [h - lower + 1 for h in c["halt_values"]]
    config = Phi3Config(
        vocab_size=tokens,
        hidden_size=size,
        intermediate_size=hidden,
        num_hidden_layers=len(layers),
        num_attention_heads=size // HEAD_DIM,
        num_key_value_heads=size // HEAD_DIM,
        hidden_act="relu",
        max_position_embeddings=context + 1,
        original_max_position_embeddings=context + 1,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": ROTARY / HEAD_DIM},
        bos_token_id=0,
        eos_token_id=halts or None,
        pad_token_id=tokens - 1,
    )
    config.dtype = torch.float64
    phi3 = Phi3ForCausalLM(config).to(torch.float64)
    phi3.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in state.items()}, strict=True)
    path = Path(path)
    phi3.save_pretrained(path)
    _tokenizer(lower, vocab, c["halt_values"]).save_pretrained(path)
    (path / "firebrand.json").write_text(json.dumps({"lower": lower, "context": context}, indent=1) + "\n")
    return path


def _prelude_mlp(gate_up, down, features, context, hidden, inverse):
    """Steps [t >= k] for k = 1..context from r = 1/(t+1), and every positional
    input of the compiled model at machine position p = t - 1, as sums of steps."""
    columns = [(1, lambda p: p), (2, lambda p: p * p)] + [
        (f["column"], lambda p, f=f: _position_feature(f, p)) for f in features
    ]
    for k in range(1, context + 1):
        threshold = (1 / k + 1 / (k + 1)) / 2  # Between r = 1/k (t = k - 1) and r = 1/(k + 1).
        slope = 4.0 * k * (k + 1)  # Gate = +2 at t = k and -2 at t = k - 1.
        a, b = 2 * (k - 1), 2 * (k - 1) + 1
        gate_up[a, 0], gate_up[a, inverse] = slope * threshold, -slope
        gate_up[b, 0], gate_up[b, inverse] = slope * threshold - 1, -slope
        gate_up[hidden + a, 0] = gate_up[hidden + b, 0] = 1
        for column, f in columns:
            delta = f(k - 1) - f(k - 2)  # Step k moves p from k - 2 to k - 1.
            down[column, a], down[column, b] = delta, -delta
    bias = 2 * context
    gate_up[bias, 0] = gate_up[hidden + bias, 0] = 1  # relu(1) * 1 = 1: the values at p = -1.
    for column, f in columns:
        down[column, bias] = f(-1)


def _certify(model, context, size, exponent):
    """Every residual value must be exact in float32 and too small to move the pinned norm."""
    c = model.config
    bounds = [abs(v) for entry in c.get("layout", []) for v in entry["range"]]
    bounds += [context, context * context, abs(c["lower"]), abs(c["upper"])]
    largest = max(bounds)
    if largest >= 2**24:
        raise ValueError(f"residual values reach {largest}: Phi-3's float32 RMSNorm is exact only below 2**24")
    if c["width"] * largest**2 >= size * 2.0 ** (2 * exponent - 25):
        raise ValueError("residual energy could move the pinned RMSNorm mean")


def _tokenizer(lower, vocab, halts):
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    words = {"<bos>": 0, **{str(lower + i): i + 1 for i in range(vocab)}, "<unk>": vocab + 1}
    tokenizer = Tokenizer(WordLevel(words, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 0)])
    tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        unk_token="<unk>",
        eos_token=str(halts[0]) if halts else None,
        pad_token="<unk>",
    )


def prompt_ids(model, tokens):
    """Hugging Face input ids for a compiled model's prompt: <bos>, then each token."""
    return [0] + [t - model.config["lower"] + 1 for t in tokens]


def tokens_from_ids(model, ids):
    """Compiled-model tokens from Hugging Face ids (dropping <bos>)."""
    return [i - 1 + model.config["lower"] for i in ids if i != 0]
