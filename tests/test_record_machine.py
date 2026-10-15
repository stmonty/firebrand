import json
from pathlib import Path

import pytest

from firebrand import Field, Record, RecordMachine, Transition, compile_machine
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var
from firebrand.machines.stack import StackSpec


def counter_definition():
    record = Record("Counter", (Field("count", 0, 7), Field("total", 0, 28), Field("address", 0, 32)))
    return RecordMachine(record, prefix_length=6, max_context=32, lower=-1, upper=32)


def counter_step(step):
    count = step.state["count"]
    return Transition(
        dict(count=(count - 1).clip(0, 7), total=(step.state["total"] + count).clip(0, 28), address=step.address),
        halt=count.equals(0),
    )


def test_definition_runs_builder_only_once_and_handles_three_field_records():
    definition = counter_definition()
    calls = []

    def build(step):
        calls.append(step)
        assert step.state["count"] is step.state["count"]
        assert set(step.state) == {"count", "total", "address"}
        return counter_step(step)

    machine = definition.lower_machine(build)
    model = compile_machine(machine, max_context=definition.max_context)
    prefix = definition.initialize(count=3, total=0, address=0)
    trace = model.generate(prefix, definition.token_budget(3))
    assert trace == prefix + [2, 3, 6, 1, 5, 9, 0, 6, 12, -1]
    assert len(calls) == 1
    assert definition.records(trace)[-1] == (12, dict(count=0, total=6, address=12))
    assert definition.records(trace, include_initial=True)[0][0] == 3


def test_prefix_and_transition_contracts():
    definition = counter_definition()
    with pytest.raises(ValueError, match="align"):
        RecordMachine(definition.record, 5, 32, -1, 32)
    with pytest.raises(ValueError, match="reserved"):
        RecordMachine(definition.record, 6, 32, -1, 32, halt_token=0)
    with pytest.raises(ValueError, match="every record field"):
        definition.lower_machine(lambda step: Transition({"count": 0}))
    with pytest.raises(TypeError, match="Transition"):
        definition.lower_machine(lambda step: 0)
    with pytest.raises(KeyError):
        definition.lower_machine(lambda step: Transition({"count": step.state["missing"]}))
    with pytest.raises(ValueError, match="fit"):
        definition.initialize([0] * 4, count=0, total=0, address=0)
    with pytest.raises(ValueError, match="alphabet"):
        definition.initialize([-1], count=0, total=0, address=0)
    with pytest.raises(ValueError, match="range"):
        definition.initialize(count=8, total=0, address=0)
    with pytest.raises(ValueError, match="budget"):
        definition.token_budget(9)
    model = compile_machine(definition.lower_machine(counter_step), max_context=32)
    with pytest.raises(ValueError, match="prefix"):
        model.generate([0, 0, 0], 0)
    lowered = definition.lower_machine(counter_step)
    assert compile_machine(lowered).config["max_context"] == 32
    with pytest.raises(ValueError, match="context_limit"):
        compile_machine(lowered, max_context=64)


def test_partial_and_invalid_halts():
    definition = counter_definition()
    prefix = definition.initialize(count=1, total=0, address=0)
    assert definition.records(prefix + [0], allow_partial=True) == []
    assert definition.records(prefix + [-1]) == []
    with pytest.raises(ValueError, match="incomplete"):
        definition.records(prefix + [0])
    with pytest.raises(ValueError, match="incomplete"):
        definition.records(prefix + [0, -1], allow_partial=True)
    with pytest.raises(ValueError, match="halt"):
        definition.records(prefix + [-1, 0, 0])
    with pytest.raises(ValueError, match="shorter"):
        definition.records([])


def test_preserves_all_captured_pre_refactor_traces():
    baseline = json.loads((Path(__file__).parent / "fixtures/protocol_baselines.json").read_text())
    stack = StackSpec(capacity=4, max_context=32, value_limit=2)
    model = compile_machine(stack.machine(), max_context=stack.max_context)
    for case in baseline["stack"]:
        program = case["program"]
        assert model.generate(stack.encode(program), stack.protocol.token_budget(len(program))) == case["trace"]

    identity = Lam(Var(0))
    constant = Lam(Lam(Var(1)))
    self_apply = Lam(App(Var(0), Var(0)))
    omega = App(self_apply, self_apply)
    terms = dict(
        identity=identity,
        apply_identity=App(identity, identity),
        unused_omega=App(App(constant, identity), omega),
        captured=App(constant, identity),
        omega=omega,
        under_lambda=Lam(omega),
    )
    spec = LambdaSpec()
    model = compile_machine(spec.machine(), max_context=spec.max_context)
    for case in baseline["lambda"]:
        prefix, _ = spec.serialize(terms[case["name"]])
        assert model.generate(prefix, spec.token_budget(16)) == case["trace"]
