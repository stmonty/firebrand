import pytest

from firebrand import compile_machine
from firebrand.machines.inspection import format_lambda_trace, inspect_lambda, inspect_records
from firebrand.machines.lambda_calculus import App, Closure, Lam, LambdaSpec, Var, validate_closed
from firebrand.machines.readback import ReadbackLimitError, format_term, readback

I = Lam(Var(0))
K = Lam(Lam(Var(1)))


def test_readback_preserves_binding_and_does_not_reduce():
    closure = Closure(Lam(App(Var(0), Var(1))), (Closure(I),))
    result = readback(closure)
    assert result == Lam(App(Var(0), I))
    validate_closed(result)
    assert format_term(result) == "λx0. (x0) (λx1. x1)"
    reducible = App(I, I)
    assert readback(Closure(Lam(Var(1)), (Closure(reducible),))) == Lam(reducible)


def test_nested_captures_use_their_own_environments():
    captured = Closure(Var(1), (Closure(K), Closure(I)))
    assert readback(Closure(Lam(Lam(Var(2))), (captured,))) == Lam(Lam(I))
    assert format_term(readback(Closure(I))) == "λx0. x0"
    with pytest.raises(ValueError, match="free variable"):
        readback(Closure(Var(0)))
    with pytest.raises(ValueError, match="open term"):
        format_term(Var(0))
    assert format_term(Lam(Var(1)), allow_open=True) == "λx0. #1"


def test_readback_is_bounded_and_rejects_cycles():
    value = Closure(I)
    for _ in range(8):
        value = Closure(App(Var(0), Var(0)), (value,))
    with pytest.raises(ReadbackLimitError):
        readback(value, max_nodes=100)
    with pytest.raises(ReadbackLimitError):
        readback(Closure(Lam(Lam(I))), max_depth=1)
    cyclic = Closure(Var(0))
    object.__setattr__(cyclic, "environment", (cyclic,))
    with pytest.raises(ValueError, match="cyclic"):
        readback(cyclic)


@pytest.fixture(scope="module")
def machine():
    spec = LambdaSpec()
    return spec, compile_machine(spec.machine())


def execute(machine, term, steps=16):
    spec, model = machine
    prefix, terms = spec.serialize(term)
    trace = model.generate(prefix, spec.token_budget(steps))
    return trace, terms, inspect_lambda(spec, trace, terms)


def test_semantic_events_and_actual_allocation_addresses(machine):
    spec, _ = machine
    trace, terms, report = execute(machine, App(I, I))
    assert [event.action for event in report.events] == ["APP", "BETA", "LOOKUP_START", "LOOKUP_RESOLVE"]
    assert [event.record_address for event in report.events] == [64, 72, 80, 88]
    assert report.events[0].allocation == "argument frame"
    assert report.events[1].allocation == "environment binding"
    assert report.events[1].semantic_reads[1] == 64
    assert report.events[-1].semantic_reads == (72,)
    assert report.events[0].changes["stack"] == (0, 64)
    assert report.result_term() == I
    assert "HALT after 4" in format_lambda_trace(report)
    assert report.records.partial_tokens == ()
    # Inspection is read-only and repeatable.
    copy = list(trace)
    assert inspect_lambda(spec, trace, terms) == report
    assert trace == copy


def test_captured_result_and_binding_walk(machine):
    _, _, report = execute(machine, App(K, I))
    assert format_term(report.result_term()) == "λx0. λx1. x1"
    _, _, walking = execute(machine, App(App(K, K), I))
    assert "LOOKUP_NEXT" in [event.action for event in walking.events]
    assert walking.result_term() == K


def test_no_result_without_an_observed_halt(machine):
    spec, model = machine
    self_apply = Lam(App(Var(0), Var(0)))
    _trace, _, report = execute(machine, App(self_apply, self_apply), steps=3)
    assert not report.records.halted
    assert len(report.events) == 3
    assert len(report.records.partial_tokens) == 1
    with pytest.raises(ValueError, match="without a halt"):
        report.result_term()
    assert "STOPPED WITHOUT HALT" in format_lambda_trace(report)
    prefix, terms = spec.serialize(I)
    untouched = inspect_lambda(spec, prefix, terms)
    assert untouched.final_state.halted and not untouched.records.halted
    with pytest.raises(ValueError):
        untouched.result_term()
    halted = inspect_lambda(spec, model.generate(prefix, 1), terms)
    assert halted.records.halted and halted.events == ()
    assert halted.result_term() == I


def test_partial_records_and_corrupt_pointers_are_not_misrepresented(machine):
    spec, _ = machine
    trace, terms, _ = execute(machine, App(I, I))
    # A self-referential closure link must fail instead of looping in the viewer.
    bad = trace.copy()
    bad[64 + spec.frame.offset("parent")] = 64
    with pytest.raises(ValueError, match="earlier"):
        inspect_lambda(spec, bad, terms)
    bad = trace.copy()
    bad[64 + spec.frame.offset("environment")] = 88
    with pytest.raises(ValueError, match="future"):
        inspect_lambda(spec, bad, terms)
    prefix, terms = spec.serialize(App(I, I))
    with pytest.raises(ValueError, match="weak-head"):
        inspect_lambda(spec, prefix + [-1], terms)
    assert len(inspect_records(spec.protocol, prefix + [1, 2]).partial_tokens) == 2
