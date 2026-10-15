import itertools
import json
from pathlib import Path

import pytest

from firebrand import Field, Record, RecordMachine, Step, Transformer, WriteLog, compile_machine
from firebrand.machines.histogram import HistogramSpec
from firebrand.machines.instruction import InstructionSpec


def test_instruction_traces_unchanged():
    # Captured before extraction; never regenerate as part of ordinary tests.
    cases = json.loads((Path(__file__).parent / "fixtures/instruction_memory_baselines.json").read_text())
    spec = InstructionSpec()
    model = compile_machine(spec.machine())
    for case in cases:
        trace = model.generate(spec.encode(case["program"]), spec.protocol.token_budget(case["steps"]))
        assert trace == case["trace"]


def test_histograms_exhaustively_and_each_write():
    spec = HistogramSpec(capacity=3, max_context=512)
    model = compile_machine(spec.machine())
    for size in range(4):
        for items in itertools.product(range(spec.cells), repeat=size):
            trace = model.generate(spec.encode(items), spec.protocol.token_budget(30))
            counts, consumed = [0] * spec.cells, 0
            for base, state in spec.protocol.records(trace):
                if state["mem_head"] == base:
                    key = items[consumed]
                    counts[key] += 1
                    assert state["mem_write_address"] == key
                    assert state["mem_write_value"] == counts[key]
                    consumed += 1
            assert consumed == len(items)
            assert spec.result(trace) == counts


def test_histogram_reload_and_partial(tmp_path):
    spec = HistogramSpec()
    model = compile_machine(spec.machine())
    model.save(tmp_path)
    prefix = spec.encode([1, 0, 1, 2, 1])
    budget = spec.protocol.token_budget(60)
    trace = Transformer.load(tmp_path).generate(prefix, budget)
    assert spec.result(trace) == [1, 3, 1]
    assert model.generate(prefix, budget, use_cache=False) == trace
    with pytest.raises(ValueError):
        spec.result(model.generate(prefix, spec.state.width * 2))


def test_memory_contracts():
    memory = WriteLog(3, 7, 128, prefix="data_")
    assert memory.initial_state() == {f.name: 0 for f in memory.fields}
    wrong = Record("Wrong", (Field("other", 0, 7),))
    definition = RecordMachine(wrong, 2, 128, -1, 128)
    with pytest.raises(ValueError, match="memory field"):
        memory.step(Step(definition), wrong)
    fields = list(memory.fields)
    fields[-1] = Field("data_parent", 0, 127)
    wrong = Record("WrongBounds", tuple(fields))
    definition = RecordMachine(wrong, 14, 128, -1, 128)
    with pytest.raises(ValueError, match="data_parent"):
        memory.step(Step(definition), wrong)
    for args in [(0, 7, 128), (1, -1, 128), (1, 7, 0)]:
        with pytest.raises(ValueError):
            WriteLog(*args)
    for items in [[3], [-1], [True], [0] * 9]:
        with pytest.raises((ValueError, TypeError)):
            HistogramSpec().encode(items)


def test_request_priority_busy_and_saturation():
    from types import SimpleNamespace

    from firebrand import constant
    from firebrand.reference import evaluate

    memory = WriteLog(2, 7, 128)
    record = Record("Memory", memory.fields)
    # One real write at address 7, containing memory[1] = 5.
    tokens = [0] * record.width + record.pack(
        head=7, searching=0, target=0, cursor=0, write_address=1, write_value=5, parent=0
    )

    def update(**state):
        initial = memory.initial_state() | state
        step = SimpleNamespace(state={k: constant(v) for k, v in initial.items()}, address=constant(14))
        result = memory.step(step, record, read=1, write=1, address=99, value=99)
        return (
            {k: evaluate(v, tokens) for k, v in result.fields.items()},
            evaluate(result.ready, tokens),
            evaluate(result.value, tokens),
        )

    fields, ready, _ = update(head=7)
    assert fields["head"] == 7  # Read wins over simultaneous write.
    assert fields["searching"] == 1 and fields["target"] == 1
    assert fields["write_value"] == 7 and ready == 0
    fields, ready, value = update(head=7, searching=1, target=1, cursor=7)
    assert ready == 1 and value == 5
    assert fields["head"] == 7 and fields["searching"] == 0  # Busy requests ignored.
    _, ready, value = update(head=7, searching=1, target=0, cursor=0)
    assert ready == 1 and value == 0
