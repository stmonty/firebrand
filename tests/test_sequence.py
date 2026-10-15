import itertools
import subprocess
import sys

import pytest

from firebrand import (
    Assign,
    Field,
    Read,
    Record,
    SequenceMachine,
    Transformer,
    Write,
    WriteLog,
    compile_machine,
)
from firebrand.machines.histogram import HistogramSpec
from firebrand.machines.scheduled_histogram import ScheduledHistogramSpec


def test_scheduled_histograms_match_manual_machine_and_reference():
    scheduled = ScheduledHistogramSpec(capacity=3, max_context=512)
    manual = HistogramSpec(capacity=3, max_context=512)
    model = compile_machine(scheduled.machine())
    other = compile_machine(manual.machine())
    for size in range(4):
        for items in itertools.product(range(3), repeat=size):
            trace = model.generate(scheduled.encode(items), scheduled.protocol.token_budget(30))
            expected_trace = other.generate(manual.encode(items), manual.protocol.token_budget(30))
            expected = [items.count(i) for i in range(3)]
            assert scheduled.result(trace) == manual.result(expected_trace) == expected
            # Every write is a logical update, independently of the schedule.
            consumed, counts = 0, [0] * 3
            previous = scheduled.protocol.record.unpack(trace, scheduled.protocol.prefix_length - scheduled.state.width)
            for base, state in scheduled.protocol.records(trace):
                if state["mem_head"] == base:
                    key = items[consumed]
                    counts[key] += 1
                    assert (state["mem_write_address"], state["mem_write_value"]) == (key, counts[key])
                    consumed += 1
                if state["mem_searching"]:
                    # No application field changes during start or pending lookup.
                    for name in ("index", "length", "count", "sequence_pc"):
                        assert state[name] == previous[name]
                previous = state
            assert consumed == len(items)


def make_sequence(operations, halt=lambda s: s["done"].equals(1)):
    state = Record("UserState", (Field("x", 0, 7), Field("y", 0, 7), Field("done", 0, 1)))
    return SequenceMachine(state, WriteLog(3, 7, 512, prefix="m_"), operations, halt=halt)


def test_dependent_reads_simultaneous_assignments_and_boundary_halt():
    seq = make_sequence(
        (
            Write(0, 1),
            Write(1, 6),
            Read("x", 0),
            Read("y", lambda s: s["x"]),
            # Both expressions observe the state before this Assign, i.e. swap.
            Assign({"x": lambda s: s["y"], "y": lambda s: s["x"], "done": 1}),
            Write(2, lambda s: s["x"]),  # Must still run after done becomes 1.
        )
    )
    model = compile_machine(seq.lower_machine())
    trace = model.generate(seq.initialize(x=0, y=0, done=0), seq.protocol.token_budget(30))
    records = seq.protocol.records(trace)
    last = records[-1][1]
    assert trace[-1] == -1 and (last["x"], last["y"], last["done"]) == (6, 1, 1)
    assert last["sequence_pc"] == 0
    assert last["m_head"] == records[-1][0] and last["m_write_value"] == 6
    # A full record budget of one stops in the body, without observing halt.
    partial = model.generate(seq.initialize(x=0, y=0, done=0), seq.protocol.token_budget(1))
    assert partial[-1] != -1


def test_initial_halt_and_pending_read_budget():
    seq = make_sequence((Read("x", 0), Assign({"done": 1})))
    model = compile_machine(seq.lower_machine())
    prefix = seq.initialize(x=3, y=4, done=1)
    assert model.generate(prefix, 100) == prefix + [-1]
    partial = model.generate(seq.initialize(x=3, y=4, done=0), seq.protocol.token_budget(1))
    r = seq.protocol.records(partial, allow_partial=True)[-1][1]
    assert (r["sequence_pc"], r["m_searching"], r["x"], r["done"]) == (0, 1, 3, 0)
    assert partial[-1] != -1


def test_one_operation_loop_and_callbacks_run_only_at_lowering(tmp_path):
    calls = []

    def increment(s):
        calls.append("build")
        return (s["x"] + 1).clip(0, 7)

    seq = make_sequence((Assign({"x": increment}),), halt=lambda s: s["x"].equals(3))
    model = compile_machine(seq.lower_machine())
    assert calls == ["build"]
    prefix = seq.initialize(x=0, y=0, done=0)
    budget = seq.protocol.token_budget(4)
    trace = model.generate(prefix, budget)
    assert [r["x"] for _, r in seq.protocol.records(trace)] == [1, 2, 3]
    assert calls == ["build"]
    assert model.generate(prefix, budget, use_cache=False) == trace
    model.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix, budget) == trace
    script = """
import sys
from firebrand import Transformer
sys.modules['firebrand.sequence'] = None
sys.modules['firebrand.memory'] = None
model = Transformer.load(sys.argv[1])
prefix = [int(v) for v in sys.argv[2].split(',')]
print(model.generate(prefix, int(sys.argv[3])))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), ",".join(map(str, prefix)), str(budget)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(trace)


@pytest.mark.parametrize(
    "operations", [(), (Read("missing", 0),), (Assign({"missing": 0}),), (Assign({}),), (object(),)]
)
def test_invalid_operations(operations):
    with pytest.raises((ValueError, TypeError)):
        make_sequence(operations)


def test_invalid_layout_and_initialization():
    memory = WriteLog(3, 7, 512)
    with pytest.raises(ValueError, match="collide"):
        SequenceMachine(Record("Bad", (Field("head", 0, 7),)), memory, (Write(0, 0),))
    with pytest.raises(ValueError, match="value range"):
        SequenceMachine(Record("Bad", (Field("x", 0, 3),)), memory, (Read("x", 0),))
    seq = make_sequence((Write(0, 0),))
    with pytest.raises(ValueError, match="payload"):
        seq.initialize([1], x=0, y=0, done=0)
    with pytest.raises(ValueError):
        seq.initialize(x=0, y=0, done=0, sequence_pc=1)
