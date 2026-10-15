from functools import lru_cache

import pytest

from firebrand import Transformer, compile_machine
from firebrand.machines.lambda_calculus import App, Closure, Lam, LambdaSpec, Var, reference_states

I = Lam(Var(0))
K = Lam(Lam(Var(1)))
SELF = Lam(App(Var(0), Var(0)))
OMEGA = App(SELF, SELF)


@pytest.fixture(scope="module")
def compiled():
    spec = LambdaSpec(max_context=256)
    return spec, compile_machine(spec.machine(), max_context=spec.max_context)


def check(compiled, term, steps=16):
    spec, model = compiled
    prefix, terms = spec.serialize(term)
    trace = model.generate(prefix, spec.token_budget(steps))
    states = spec.states(trace, terms)
    expected = reference_states(term, steps)
    assert states == expected  # Every completed transition, including closures.
    assert (trace[-1] == -1) == expected[-1].halted
    assert trace[: len(prefix)] == prefix
    return states, trace


@pytest.mark.parametrize(
    "term",
    [
        I,
        App(I, I),
        App(App(K, I), OMEGA),  # Call-by-name must not evaluate the unused Omega.
        App(Lam(Lam(Var(1))), I),  # Result closure retains an environment.
        App(Lam(App(Var(0), I)), I),  # Resolve a closure then apply it.
        App(App(Lam(Lam(Var(1))), K), I),  # Walk past one lexical binding.
    ],
)
def test_transitions_and_lexical_environments(compiled, term):
    states, _ = check(compiled, term)
    assert states[-1].halted


def test_result_is_a_closure_not_just_a_term(compiled):
    states, _ = check(compiled, App(K, I))
    result = states[-1]
    assert result.term == Lam(Var(1))
    assert result.environment == (Closure(I),)


def test_divergence_stops_at_budget_without_claiming_normalization(compiled):
    states, trace = check(compiled, OMEGA, steps=16)
    assert len(states) == 17
    assert trace[-1] != -1
    assert not states[-1].halted


def test_does_not_reduce_under_lambda(compiled):
    term = Lam(OMEGA)
    states, trace = check(compiled, term)
    assert len(states) == 1
    assert states[-1].term == term
    assert trace[-1] == -1


def test_zero_budget_and_context_boundary(compiled):
    states, trace = check(compiled, I, steps=0)
    assert states[-1].halted and trace[-1] == -1
    states, trace = check(compiled, App(I, I), steps=0)
    assert len(states) == 1 and trace[-1] != -1
    # The final permitted forward pass can observe halt or emit an incomplete
    # next record. It must not be decoded as an additional completed transition.
    spec, _ = compiled
    budget = (spec.max_context - spec.prefix_length) // 8
    states, trace = check(compiled, OMEGA, steps=budget)
    assert len(states) == budget + 1
    assert len(trace) == spec.max_context + 1


@lru_cache(None)
def closed_terms(size, depth=0):
    if size == 1:
        return tuple(Var(index) for index in range(depth))
    terms = [Lam(body) for body in closed_terms(size - 1, depth + 1)]
    for left_size in range(1, size - 1):
        for left in closed_terms(left_size, depth):
            for right in closed_terms(size - 1 - left_size, depth):
                terms.append(App(left, right))
    return tuple(terms)


def test_all_closed_terms_through_six_nodes(compiled):
    for size in range(1, 7):
        for term in closed_terms(size):
            check(compiled, term, steps=16)


def test_checkpoint_and_program_independence(compiled, tmp_path):
    spec, model = compiled
    model.save(tmp_path)
    restored = Transformer.load(tmp_path)
    for term in (App(I, I), App(App(K, I), OMEGA)):
        check((spec, restored), term)


def test_input_and_budget_contracts():
    spec = LambdaSpec()
    with pytest.raises(ValueError, match="free variable"):
        spec.serialize(Var(0))
    with pytest.raises(ValueError, match="free variable"):
        spec.serialize(Lam(Var(1)))
    with pytest.raises(ValueError, match="capacity"):
        LambdaSpec(node_capacity=8).serialize(Lam(Lam(Lam(Lam(Lam(Lam(Lam(Lam(Lam(Var(0)))))))))))
    with pytest.raises(ValueError, match="budget"):
        spec.token_budget(100)
    with pytest.raises(ValueError, match="multiple"):
        LambdaSpec(node_capacity=3)
