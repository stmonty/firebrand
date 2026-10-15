import itertools

import pytest

from firebrand import Machine, Transformer, compile_machine, history, select
from firebrand.machines.stack import POP, PUSH, StackSpec, reference_stack
from firebrand.records import Field, Record
from firebrand.reference import run


@pytest.mark.parametrize(
    "expr, lower, upper",
    [
        (history().equals(-2), -4, 4),
        (history() % 3, 0, 9),
        (select(history().equals(0), -history(1), history(1)), -4, 4),
    ],
)
def test_control_primitives(expr, lower, upper):
    machine = Machine(expr, lower, upper)
    model = compile_machine(machine)
    for prefix in itertools.product(range(lower, upper + 1), repeat=machine.required_prefix):
        assert model.generate(list(prefix), 1) == run(machine, list(prefix), 1)


def test_mod_and_selection_contracts():
    with pytest.raises(ValueError, match="nonnegative"):
        compile_machine(Machine(history() % 2, -2, 2))
    with pytest.raises(ValueError, match="condition"):
        compile_machine(Machine(select(history(), 1, 0), 0, 2))
    with pytest.raises(ValueError, match="positive"):
        history() % 0


def test_record_layouts():
    pair = Record("Pair", (Field("value", -2, 2), Field("parent", 0, 10)))
    assert pair.pack(value=-1, parent=4) == [-1, 4]
    assert pair.unpack([9, -1, 4], 1) == {"value": -1, "parent": 4}
    model = compile_machine(Machine(pair.read(history(), "parent"), 0, 10))
    assert model.generate([2, 4, 0], 1)[-1] == 4
    with pytest.raises(ValueError):
        pair.pack(value=3, parent=0)
    with pytest.raises(ValueError):
        pair.unpack([1])
    with pytest.raises(KeyError):
        pair.read(0, "missing")


def assert_stack(spec, model, program):
    prefix = spec.encode(program)
    trace = model.generate(prefix, len(program) * 4 + 1)
    assert trace[-1] == -1
    assert trace[: len(prefix)] == prefix
    records = spec.decode(trace)
    assert len(records) == len(program)
    for i, (record, (expected, stack)) in enumerate(zip(records, reference_stack(program))):
        assert record["value"] == expected
        address = record["top"]
        actual = []
        while address:
            # Every live frame was emitted before or within this instruction.
            assert spec.prefix_length <= address <= spec.prefix_length + 4 * i
            frame = spec.frame.unpack(trace, address)
            actual.append(frame["value"])
            assert frame["parent"] < address
            address = frame["parent"]
        assert actual == stack[::-1]
    return trace


def test_all_short_programs_with_one_model():
    spec = StackSpec(capacity=4, max_context=32, value_limit=2)
    model = compile_machine(spec.machine(), max_context=spec.max_context)
    commands = [(PUSH, 1), (PUSH, 2), (POP, 0)]
    for length in range(4):
        for program in itertools.product(commands, repeat=length):
            assert_stack(spec, model, program)


def test_exported_weights_allocate_and_pop_frames(tmp_path):
    spec = StackSpec()
    machine = spec.machine()
    model = compile_machine(machine, max_context=spec.max_context)
    model.save(tmp_path)
    loaded = Transformer.load(tmp_path)
    program = [(PUSH, 2), (PUSH, 5), (POP, 0), (PUSH, 7), (POP, 0), (POP, 0), (POP, 0)]
    trace = assert_stack(spec, loaded, program)
    assert trace == run(machine, spec.encode(program), 4 * len(program) + 1)
    # The first two frames remain readable after all their values were popped.
    assert spec.frame.unpack(trace, spec.prefix_length)["value"] == 2
    assert spec.frame.unpack(trace, spec.prefix_length + 4)["value"] == 5


def test_stack_limits_and_partial_records():
    spec = StackSpec(capacity=2, max_context=16)
    with pytest.raises(ValueError):
        spec.encode([(PUSH, 1), (POP, 0)])
    with pytest.raises(ValueError):
        spec.encode([(PUSH, -1)])
    with pytest.raises(ValueError):
        spec.decode(spec.encode([]) + [1])
    with pytest.raises(ValueError):
        StackSpec(capacity=3)
