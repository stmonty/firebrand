import math

import pytest

from firebrand import (
    Assign,
    Branch,
    Field,
    Halt,
    Jump,
    Label,
    Read,
    Record,
    SequenceMachine,
    Transformer,
    Write,
    WriteLog,
    compile_machine,
    read_at,
)
from firebrand.machines.gcd import gcd_machine


def test_gcd_all_small_inputs_and_every_reduction():
    program = gcd_machine(value_limit=7, max_context=512)
    model = compile_machine(program.lower_machine())
    for a in range(8):
        for b in range(8):
            trace = model.generate(program.initialize(a=a, b=b, result=0), program.protocol.token_budget(44))
            records = program.protocol.records(trace)
            assert trace[-1] == -1
            assert records[-1][1]["result"] == math.gcd(a, b)
            expected, x, y = [], a, b
            while x and y and x != y:
                if x > y:
                    x -= y
                else:
                    y -= x
                expected.append((x, y))
            # These operations perform subtraction; compare every resulting pair.
            observed, previous_pc = [], 0
            for _, state in records:
                if previous_pc in (program.labels["reduce_a"], program.labels["reduce_b"]):
                    observed.append((state["a"], state["b"]))
                previous_pc = state["sequence_pc"]
            assert observed == expected


def program_with(ops):
    return SequenceMachine(Record("BranchTest", (Field("x", 0, 7),)), WriteLog(2, 7, 256), ops)


def test_branch_after_pending_read_skips_unselected_memory_effects(tmp_path):
    program = program_with(
        (
            Write(0, lambda s: s["x"]),
            Read("x", 0),
            Branch(lambda s: s["x"].equals(0), "zero", "nonzero"),
            Label("zero"),
            Write(1, 2),
            Jump("end"),
            Label("nonzero"),
            Write(1, 6),
            Label("end"),
            Halt(),
            Assign({"x": 7}),  # Unreachable after the explicit Halt.
        )
    )
    model = compile_machine(program.lower_machine())
    model.save(tmp_path)
    for x, expected in ((0, 2), (3, 6)):
        prefix = program.initialize(x=x)
        trace = Transformer.load(tmp_path).generate(prefix, program.protocol.token_budget(15))
        assert trace == model.generate(prefix, program.protocol.token_budget(15), use_cache=False)
        assert trace[-1] == -1
        records = program.protocol.records(trace)
        writes = [(s["write_address"], s["write_value"]) for base, s in records if s["head"] == base]
        assert writes == [(0, x), (1, expected)]
        assert records[-1][1]["x"] == x
        pending = [s for _, s in records if s["searching"]]
        assert len(pending) == 1 and pending[0]["sequence_pc"] == 1


def test_labels_are_free_aliases_and_loops_respect_budget():
    program = program_with((Label("start"), Label("again"), Jump("again")))
    assert program.labels == {"start": 0, "again": 0}
    assert len(program.executable) == 1
    model = compile_machine(program.lower_machine())
    trace = model.generate(program.initialize(x=2), program.protocol.token_budget(4))
    records = program.protocol.records(trace, allow_partial=True)
    assert len(records) == 4 and trace[-1] != -1
    assert all(s["x"] == 2 and s["sequence_pc"] == 0 for _, s in records)


def test_immediate_halt_emits_no_state_record():
    program = program_with((Label("stop"), Halt()))
    model = compile_machine(program.lower_machine())
    prefix = program.initialize(x=4)
    assert model.generate(prefix, 1) == prefix + [-1]


@pytest.mark.parametrize(
    "ops",
    [
        (Jump("missing"),),
        (Branch(1, "yes", "no"), Label("yes"), Halt()),
        (Label("same"), Assign({"x": 0}), Label("same"), Halt()),
        (Assign({"x": 0}), Label("trailing")),
        (Label(""), Halt()),
        (Label(3), Halt()),
        (Jump(3), Halt()),
        (Label("only"),),
    ],
)
def test_invalid_control_flow(ops):
    with pytest.raises(ValueError):
        program_with(ops)


def test_branch_condition_must_be_boolean():
    program = program_with((Branch(lambda s: s["x"], "end", "end"), Label("end"), Halt()))
    with pytest.raises(ValueError, match="condition"):
        compile_machine(program.lower_machine())


def test_branching_does_not_make_raw_history_reads_lazy():
    program = program_with((Jump("end"), Assign({"x": read_at(200)}), Label("end"), Halt()))
    model = compile_machine(program.lower_machine())
    with pytest.raises(ValueError, match="address"):
        model.generate(program.initialize(x=0), 1)
