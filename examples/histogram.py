"""Count categories with the compiled write-log memory (no instruction interpreter)."""

from firebrand import compile_machine
from firebrand.machines.histogram import HistogramSpec


def main():
    spec = HistogramSpec()
    model = compile_machine(spec.machine())
    categories = [1, 0, 1, 2, 1]
    trace = model.generate(spec.encode(categories), spec.protocol.token_budget(60))
    print("Categories:", categories)
    print("Counts:", spec.result(trace))
    assert spec.result(trace) == [1, 3, 1]


if __name__ == "__main__":
    main()
