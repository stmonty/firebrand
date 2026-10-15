import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from firebrand import Machine, Transformer, compile_machine, constant, history
from firebrand.ir import nodes
from firebrand.reference import evaluate, run


@pytest.mark.parametrize(
    "expr",
    [
        history(),
        -history(),
        (history() - 1).relu(),
        (2 * history() + history(1) - 3).clip(-4, 4),
        (history() + history()).clip(-4, 4),
        constant(-2),
    ],
)
def test_exhaustive_outputs_and_every_intermediate(expr):
    machine = Machine(expr, -4, 4)
    # ReLU countdown remains in [-4,4], and all other expressions are bounded.
    model = compile_machine(machine)
    for prefix in itertools.product(range(-4, 5), repeat=machine.required_prefix):
        logits, probes = model.logits(list(prefix), probes=True)
        assert int(np.argmax(logits)) - 4 == evaluate(expr, list(prefix))
        for node, description in zip(nodes(expr), model.config["layout"]):
            assert probes[description["stage"]][description["slot"]] == evaluate(node, list(prefix))


def test_independent_known_sequences():
    countdown = compile_machine(Machine((history() - 1).relu(), 0, 31, (0,)))
    assert countdown.generate([5]) == [5, 4, 3, 2, 1, 0]
    fib = compile_machine(Machine((history() + history(1)).clip(0, 31), 0, 31, (31,)))
    assert fib.generate([0, 1]) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 31]


def test_long_trace_and_context_limit():
    machine = Machine(-history(), -3, 3)
    model = compile_machine(machine, max_context=256)
    assert model.generate([3], 256) == run(machine, [3], 256)
    with pytest.raises(ValueError, match="context"):
        model.generate([3], 257)


def test_history_head_selects_requested_position():
    model = compile_machine(Machine(history(2), -4, 4))
    assert model.generate([1, 2, 3, -4], 4) == [1, 2, 3, -4, 2, 3, -4, 2]
    # Exercise addressing at the declared maximum position.
    prefix = [0] * 4096
    prefix[-3:] = [-4, 1, 4]
    assert model.generate(prefix, 1)[-1] == -4


def test_weight_mutation_changes_execution():
    model = compile_machine(Machine(-history(), -3, 3))
    assert model.generate([3], 1)[-1] == -3
    model.weights["ff_0_out"] *= -1
    assert model.generate([3], 1)[-1] == 3


def test_checkpoint_executes_without_source_evaluator(tmp_path):
    model = compile_machine(Machine((history() - 1).relu(), 0, 7, (0,)))
    model.save(tmp_path)
    restored = Transformer.load(tmp_path)
    assert restored.generate([4]) == [4, 3, 2, 1, 0]
    script = """
import sys
sys.modules['firebrand.reference'] = None
from firebrand.runtime import Transformer
print(Transformer.load(sys.argv[1]).generate([4]))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "[4, 3, 2, 1, 0]"
    assert "next_value" not in json.loads((tmp_path / "config.json").read_text())


def test_invalid_contracts():
    with pytest.raises(ValueError, match="output range"):
        compile_machine(Machine(history() + 1, 0, 7))
    with pytest.raises(ValueError, match="coefficient"):
        compile_machine(Machine(constant(2**27).clip(0, 7), 0, 7))
    with pytest.raises(ValueError, match="max_context"):
        compile_machine(Machine(history(2), 0, 7), max_context=2)
    with pytest.raises(TypeError):
        history() * 0.5
    model = compile_machine(Machine(history(1), 0, 7))
    for prefix in ([], [1], [1, 8], [1, 1.5], [True, 1]):
        with pytest.raises(ValueError):
            model.generate(prefix, 0)


def test_signed_clip_and_large_integer_decoding():
    expr = (history() * -2).clip(-100001, -100000)
    machine = Machine(expr, -100001, -100000)
    model = compile_machine(machine)
    assert model.generate([-100001], 5) == run(machine, [-100001], 5)
