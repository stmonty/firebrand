"""Cached execution must preserve intermediate vectors and runtime contracts."""

import numpy as np
import pytest

from firebrand import Machine, compile_machine, constant, history, read_at
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var
from firebrand.machines.stack import POP, PUSH, StackSpec
from firebrand.runtime import _KVCache


@pytest.mark.parametrize(
    "expr,prefix",
    [
        (-history(), [3]),
        (constant(2), [1]),
        (read_at(read_at(history())), [0, 0, 1, 2, 3]),
    ],
)
def test_every_cached_stage_matches_reference(expr, prefix):
    model = compile_machine(Machine(expr, -7, 7), max_context=32)
    tokens = prefix.copy()
    cache = _KVCache(model, tokens, 32)
    for _ in range(12):
        actual, stages = cache.logits(probes=True)
        expected, reference = model.logits(tokens, probes=True)
        np.testing.assert_array_equal(actual, expected)
        assert stages.keys() == reference.keys()
        for name in stages:
            np.testing.assert_array_equal(stages[name], reference[name])
        token = int(actual.argmax()) + model.config["lower"]
        tokens.append(token)
        cache.append(token)


def test_applications_and_reloaded_weights(tmp_path):
    stack = StackSpec()
    lam = LambdaSpec()
    identity = Lam(Var(0))
    for spec, prefix, budget in [
        (stack, stack.encode([(PUSH, 3), (PUSH, 5), (POP, 0)]), 13),
        (lam, lam.serialize(App(identity, identity))[0], lam.token_budget(16)),
    ]:
        model = compile_machine(spec.machine())
        expected = model.generate(prefix, budget, use_cache=False)
        assert model.generate(prefix, budget) == expected
        model.save(tmp_path)
        assert type(model).load(tmp_path).generate(prefix, budget) == expected


@pytest.mark.parametrize("use_cache", [False, True])
def test_boundaries_and_invalid_addresses(use_cache):
    model = compile_machine(Machine(-history(), -3, 3), max_context=8)
    assert len(model.generate([3], 8, use_cache=use_cache)) == 9
    with pytest.raises(ValueError, match="context"):
        model.generate([3], 9, use_cache=use_cache)
    with pytest.raises(ValueError, match="alphabet"):
        model.generate([7], 0, use_cache=use_cache)
    pointer = compile_machine(Machine(read_at(history()), 0, 7, (0,)))
    assert pointer.generate([7], 0, use_cache=use_cache) == [7]
    assert pointer.generate([0], use_cache=use_cache) == [0]
    with pytest.raises(ValueError, match="address"):
        pointer.generate([7], 1, use_cache=use_cache)


def test_cache_is_fresh_and_projects_only_new_tokens(monkeypatch):
    model = compile_machine(Machine(-history(), -3, 3))
    project = model._project
    rows = []

    def count(x, index):
        rows.append(len(x))
        return project(x, index)

    monkeypatch.setattr(model, "_project", count)
    model.generate([1, 2, 3], 4)
    assert rows == [3, 1, 1, 1]
    rows.clear()
    model.generate([1, 2, 3], 4, use_cache=False)
    assert rows == [3, 4, 5, 6]
    assert model.generate([-2], 1) == [-2, 2]
    model.weights["ff_0_out"] *= -1
    assert model.generate([-2], 1) == [-2, -2]
