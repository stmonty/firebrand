"""Compare record costs before and after composing dependent assignments."""

from firebrand import (
    Assign,
    Branch,
    Field,
    Halt,
    Label,
    Record,
    SequenceMachine,
    WriteLog,
    compile_machine,
    fuse_assignments,
)


def example():
    return SequenceMachine(
        Record("Sum", (Field("count", 0, 3), Field("total", 0, 15), Field("doubled", 0, 15))),
        WriteLog(1, 15, 256),
        (
            Label("loop"),
            Assign({"total": lambda s: (s["total"] + s["count"]).clip(0, 15)}),
            Assign({"doubled": lambda s: (s["total"] * 2).clip(0, 15)}),
            Assign({"count": lambda s: (s["count"] - 1).clip(0, 3)}),
            Branch(lambda s: s["count"].equals(0), "end", "loop"),
            Label("end"),
            Halt(),
        ),
    )


def main():
    original = example()
    for name, program in (("original", original), ("fused", fuse_assignments(original))):
        model = compile_machine(program.lower_machine())
        prefix = program.initialize(count=3, total=0, doubled=0)
        trace = model.generate(prefix, program.protocol.token_budget(12))
        records = program.protocol.records(trace)
        assert trace[-1] == -1
        final = records[-1][1]
        assert (final["count"], final["total"], final["doubled"]) == (0, 6, 12)
        print(
            f"{name}: {len(program.executable)} operations, {len(records)} microsteps, "
            f"{len(trace) - len(prefix)} generated tokens, "
            f"{sum(w.size for w in model.weights.values()):,} parameters"
        )


if __name__ == "__main__":
    main()
