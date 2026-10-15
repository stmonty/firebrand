import itertools
from dataclasses import replace

import pytest

from firebrand import (
    Assign,
    Branch,
    Field,
    Halt,
    Label,
    Read,
    Record,
    SequenceMachine,
    Transformer,
    Write,
    WriteLog,
    compile_machine,
    fuse_assignments,
    history,
    position,
    read_at,
)


def sequence(operations, halt=0):
    return SequenceMachine(
        Record("Fusion", (Field("x", 0, 7), Field("y", 0, 7), Field("done", 0, 1))),
        WriteLog(2, 7, 512),
        operations,
        halt=halt,
    )


def run(program, model, **state):
    trace = model.generate(program.initialize(**state), program.protocol.token_budget(30))
    assert trace[-1] == -1
    return trace, program.protocol.records(trace)


def test_fusion_preserves_sequential_dependencies_and_simultaneous_assignments():
    original = sequence(
        (
            Assign({"x": lambda s: s["y"], "y": lambda s: s["x"]}),
            Assign({"x": lambda s: (s["x"] + s["y"]).clip(0, 7)}),
            Assign({"y": lambda s: (s["x"] + 1).clip(0, 7)}),
            Halt(),
        )
    )
    fused = fuse_assignments(original)
    assert len(original.executable) == 4 and len(fused.executable) == 2
    models = [compile_machine(p.lower_machine()) for p in (original, fused)]
    for x, y in itertools.product(range(8), repeat=2):
        finals = []
        for p, model in zip((original, fused), models):
            _, records = run(p, model, x=x, y=y, done=0)
            finals.append((records[-1][1]["x"], records[-1][1]["y"]))
        total = min(7, x + y)
        assert finals == [(total, min(7, total + 1))] * 2


def test_labels_preserve_entry_points_and_branch_paths():
    original = sequence(
        (
            Branch(lambda s: s["x"].equals(0), "middle", "start"),
            Label("start"),
            Assign({"y": 2}),
            Label("middle"),
            Label("alias"),
            Assign({"x": lambda s: s["y"]}),
            Assign({"y": lambda s: (s["x"] + 1).clip(0, 7)}),
            Halt(),
        )
    )
    fused = fuse_assignments(original)
    assert fused.labels == {"start": 1, "middle": 2, "alias": 2}
    for p in (original, fused):
        model = compile_machine(p.lower_machine())
        for x, expected in ((0, (5, 6)), (1, (2, 3))):
            _, records = run(p, model, x=x, y=5, done=0)
            assert (records[-1][1]["x"], records[-1][1]["y"]) == expected


def test_memory_effects_read_waits_and_exported_execution(tmp_path):
    original = sequence(
        (
            Assign({"x": 3}),
            Assign({"y": lambda s: s["x"]}),
            Write(0, lambda s: s["y"]),
            Write(0, 0),
            Read("x", 0),
            Assign({"y": lambda s: (s["x"] + 1).clip(0, 7)}),
            Assign({"done": 1}),
            Write(1, lambda s: s["y"]),
            Halt(),
        )
    )
    fused = fuse_assignments(original)
    assert [type(op) for op in fused.executable] == [Assign, Write, Write, Read, Assign, Write, Halt]
    for p in (original, fused):
        model = compile_machine(p.lower_machine())
        trace, records = run(p, model, x=7, y=0, done=0)
        writes = [(s["write_address"], s["write_value"]) for base, s in records if s["head"] == base]
        assert writes == [(0, 3), (0, 0), (1, 1)]
        assert (records[-1][1]["x"], records[-1][1]["y"], records[-1][1]["done"]) == (0, 1, 1)
        assert any(s["searching"] for _, s in records)
        model.save(tmp_path)
        assert Transformer.load(tmp_path).generate(p.initialize(x=7, y=0, done=0), p.protocol.token_budget(30)) == trace
        assert model.generate(p.initialize(x=7, y=0, done=0), p.protocol.token_budget(30), use_cache=False) == trace


def test_body_boundary_halt_and_budget_semantics():
    original = sequence((Assign({"done": 1}), Assign({"x": 6})), halt=lambda s: s["done"].equals(1))
    fused = fuse_assignments(original)
    assert len(fused.executable) == 1
    for p, steps in ((original, 2), (fused, 1)):
        model = compile_machine(p.lower_machine())
        trace, records = run(p, model, x=0, y=0, done=0)
        assert len(records) == steps and records[-1][1]["x"] == 6
        assert len(trace) - p.protocol.prefix_length == steps * p.record.width + 1
        initial = p.initialize(x=0, y=0, done=1)
        assert model.generate(initial, 1) == initial + [-1]
        partial = model.generate(p.initialize(x=0, y=0, done=0), p.record.width - 1)
        assert partial[-1] != -1 and p.protocol.records(partial, allow_partial=True) == []


def test_backward_jump_and_iteration_state_match():
    original = sequence(
        (
            Label("loop"),
            Assign({"y": lambda s: (s["y"] + s["x"]).clip(0, 7)}),
            Assign({"x": lambda s: (s["x"] - 1).clip(0, 7)}),
            Branch(lambda s: s["x"].equals(0), "end", "loop"),
            Label("end"),
            Halt(),
        )
    )
    for p in (original, fuse_assignments(original)):
        model = compile_machine(p.lower_machine())
        for x in range(1, 5):
            _, records = run(p, model, x=x, y=0, done=0)
            # A completed branch marks a shared semantic boundary.
            observed = [(s["x"], s["y"]) for _, s in records if s["sequence_pc"] in (p.labels["loop"], p.labels["end"])]
            expected, total = [], 0
            for count in range(x, 0, -1):
                total = min(7, total + count)
                expected.append((count - 1, total))
            assert observed == expected


@pytest.mark.parametrize("raw", [history(), position(), read_at(0)])
@pytest.mark.parametrize("place", ["assignment", "read", "write", "branch", "halt"])
def test_reject_physical_trace_dependencies_anywhere(raw, place):
    ops = [Assign({"x": 0}), Assign({"y": 1}), Halt()]
    halt = 0
    if place == "assignment":
        ops.append(Assign({"x": raw.clip(0, 7)}))  # Even unreachable code.
    elif place == "read":
        ops.append(Read("x", raw))
    elif place == "write":
        ops.append(Write(0, raw))
    elif place == "branch":
        ops.extend((Label("test"), Branch(raw.equals(0), "test", "test")))
    else:
        halt = raw.equals(0)
    with pytest.raises(ValueError, match="state-only"):
        fuse_assignments(sequence(ops, halt))


def test_reject_unproved_intermediate_field_contract_even_if_overwritten():
    original = sequence((Assign({"x": 8}), Assign({"x": 0}), Halt()))
    with pytest.raises(ValueError, match="field range"):
        fuse_assignments(original)
    with pytest.raises(ValueError, match="field range"):
        fuse_assignments(replace(original, operations=(Assign({"x": lambda s: s["x"] + 1}), Halt())))


def test_callbacks_build_once_and_never_run_during_inference():
    calls = []

    def builder(state):
        calls.append("build")
        return (state["x"] + 1).clip(0, 7)

    original = sequence((Assign({"x": builder}), Assign({"y": builder}), Halt()))
    fused = fuse_assignments(original)
    assert calls == ["build", "build"]
    model = compile_machine(fused.lower_machine())
    _, records = run(fused, model, x=0, y=0, done=0)
    assert calls == ["build", "build"]
    assert (records[-1][1]["x"], records[-1][1]["y"]) == (1, 2)


def test_signed_expressions_and_modulo():
    original = SequenceMachine(
        Record("Signed", (Field("x", -4, -2), Field("y", 0, 7))),
        WriteLog(1, 7, 128),
        (
            Assign({"y": lambda s: (-s["x"]).relu() % 3}),
            Assign({"x": lambda s: (-s["y"] - 2).clip(-4, -2)}),
            Halt(),
        ),
    )
    for p in (original, fuse_assignments(original)):
        model = compile_machine(p.lower_machine())
        for x in range(-4, -1):
            trace = model.generate(p.initialize(x=x, y=0), p.protocol.token_budget(3))
            assert trace[-1] == -1
            state = p.protocol.records(trace)[-1][1]
            assert (state["x"], state["y"]) == (-((-x) % 3) - 2, (-x) % 3)
