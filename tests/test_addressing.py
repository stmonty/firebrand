import itertools

import numpy as np
import pytest

from firebrand import Machine, Transformer, compile_machine, history, position, read_at
from firebrand.ir import nodes
from firebrand.reference import evaluate, run


@pytest.mark.parametrize(
    "expression",
    [
        read_at(history()),
        read_at(read_at(history())),
        read_at((history() + 1).clip(0, 3)),
        (read_at(history()) + read_at(history(1))).clip(0, 3),
    ],
)
def test_exhaustive_pointer_programs(expression):
    model = compile_machine(Machine(expression, 0, 3), max_context=16)
    for prefix in itertools.product(range(4), repeat=4):
        logits, probes = model.logits(list(prefix), probes=True)
        assert np.argmax(logits) == evaluate(expression, list(prefix))
        for node, entry in zip(nodes(expression), model.config["layout"]):
            assert probes[entry["stage"]][entry["slot"]] == evaluate(node, list(prefix))


def test_dependent_queries_and_weight_only_roundtrip(tmp_path):
    expr = read_at(read_at(history()))
    machine = Machine(expr, 0, 7, (0,))
    model = compile_machine(machine)
    prefix = [0, 0, 1, 2, 3, 4, 5]
    assert model.generate(prefix)[len(prefix) :] == [3, 1, 0]
    assert model.generate(prefix) == run(machine, prefix)
    model.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix) == model.generate(prefix)
    assert [s["kind"] for s in model.config["schedule"]] == ["attention"] * 3
    # Force the final head to query position zero instead. The answer changes
    # solely through weights; the runtime has no pointer-chain interpreter.
    model.weights["head_2_q"][:, 0] = 0
    assert model.generate(prefix, 1)[-1] == 0


def test_position_arithmetic_before_lookup():
    expr = read_at(position() - 1)
    machine = Machine(expr, -4, 4)
    model = compile_machine(machine, max_context=32)
    assert model.generate([3, -2, 4], 8) == run(machine, [3, -2, 4], 8)
    assert model.generate([3, -2, 4], 2) == [3, -2, 4, -2, 4]
    with pytest.raises(ValueError, match="address"):
        model.logits([3])


def test_invalid_addresses_do_not_silently_select_nearest_position():
    expr = read_at(history())
    machine = Machine(expr, -2, 7, (0,))
    model = compile_machine(machine, max_context=16)
    for prefix in ([-1], [7], [0, -2], [0, 3]):
        with pytest.raises(ValueError, match="address"):
            model.logits(prefix)
        with pytest.raises(ValueError, match="address"):
            evaluate(expr, prefix)
    assert model.generate([7], 0) == [7]  # No transition means no dereference.
    assert model.generate([0]) == [0]  # Halting avoids a transition too.
    with pytest.raises(ValueError, match="address"):
        compile_machine(Machine(read_at(-1), 0, 7))
    with pytest.raises(ValueError, match="address"):
        compile_machine(Machine(read_at(16), 0, 7), max_context=16)


def test_read_current_position_and_context_boundary():
    model = compile_machine(Machine(read_at(position()), -4, 4), max_context=32)
    assert model.generate([1, -3], 2) == [1, -3, -3, -3]
    assert model.generate([0] * 31 + [4], 1)[-1] == 4


def test_matches_full_causal_matrix_execution():
    """Validate the optimized last-query decoder against full per-layer tensors."""
    expr = read_at((read_at(history()) + 1).clip(0, 3)) + read_at(history(1).clip(0, 3))
    model = compile_machine(Machine(expr.clip(0, 3), 0, 3))
    tokens = [3, 0, 2, 1]
    w = model.weights
    x = w["embedding"][tokens].copy()
    p = np.arange(len(tokens))
    x[:, 1], x[:, 2] = p, p**2
    _, probes = model.logits(tokens, probes=True)
    for step in model.config["schedule"]:
        if step["kind"] == "attention":
            update = 0
            for i in step.get("heads", [step.get("index")]):
                q, k, v = (x @ w[f"head_{i}_{part}"] for part in ("q", "k", "v"))
                scores = q @ k.T
                scores[np.triu_indices(len(tokens), k=1)] = -np.inf
                chosen = scores.argmax(axis=1)
                update = update + v[chosen] @ w[f"head_{i}_o"]
            x = x + update
            stage = step.get("name", f"head_{i}")
        else:
            i = step["index"]
            x = x + np.maximum(0, x @ w[f"ff_{i}_in"]) @ w[f"ff_{i}_out"]
            stage = step.get("name", f"ff_{i}")
        np.testing.assert_array_equal(x[-1], probes[stage])
    np.testing.assert_array_equal(x[-1] @ w["unembedding"], model.logits(tokens))


def test_version_one_checkpoint_compatibility(tmp_path):
    model = compile_machine(Machine((history() - 1).relu(), 0, 7, (0,)))
    # This graph's old-format execution order was all heads, then all FFNs.
    model.config["format_version"] = 1
    model.config.pop("schedule")
    model.config.pop("addressing")
    for entry in model.config["layout"]:
        if entry["operation"] == "history":
            entry["stage"] = "attention"
    model.save(tmp_path)
    loaded = Transformer.load(tmp_path)
    assert loaded.generate([4]) == [4, 3, 2, 1, 0]
