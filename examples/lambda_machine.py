"""Run several closed lambda terms using the same exported transformer weights."""

import argparse

from firebrand import Transformer, compile_machine
from firebrand.machines.inspection import format_lambda_trace, inspect_lambda
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var, reference_states
from firebrand.machines.readback import format_term


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="show semantic transitions and record addresses")
    options = parser.parse_args()
    identity = Lam(Var(0))
    constant = Lam(Lam(Var(1)))
    self_apply = Lam(App(Var(0), Var(0)))
    omega = App(self_apply, self_apply)
    spec = LambdaSpec()
    model = compile_machine(spec.machine(), max_context=spec.max_context)
    model.save("artifacts/lambda")
    model = Transformer.load("artifacts/lambda")
    for name, term in (
        ("I I", App(identity, identity)),
        ("K I Omega", App(App(constant, identity), omega)),
        ("K I (closure result)", App(constant, identity)),
        ("Omega", omega),
    ):
        prefix, terms = spec.serialize(term)
        trace = model.generate(prefix, spec.token_budget(16))
        states = spec.states(trace, terms)
        assert states == reference_states(term, 16)
        status = "weak-head form" if trace[-1] == -1 else "step budget exhausted"
        print(f"{name}: {status}; {len(states) - 1} completed transitions")
        report = inspect_lambda(spec, trace, terms)
        if report.records.halted:
            print("  result:", format_term(report.result_term()))
        if options.trace:
            print(format_lambda_trace(report))


if __name__ == "__main__":
    main()
