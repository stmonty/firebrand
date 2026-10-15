"""Read/compute/write sequencing without application-defined waiting states."""

from firebrand import compile_machine
from firebrand.machines.histogram import HistogramSpec
from firebrand.machines.scheduled_histogram import ScheduledHistogramSpec


def main():
    items = [1, 0, 1, 2, 1]
    for spec in (HistogramSpec(), ScheduledHistogramSpec()):
        model = compile_machine(spec.machine())
        trace = model.generate(spec.encode(items), spec.protocol.token_budget(60))
        print(type(spec).__name__, spec.result(trace), "microsteps:", len(spec.protocol.records(trace)))
        assert spec.result(trace) == [1, 3, 1]


if __name__ == "__main__":
    main()
