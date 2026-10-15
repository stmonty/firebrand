"""Declare a record transition without writing its scalar emission protocol."""

from firebrand import Field, Record, RecordMachine, Transition, compile_machine


def main():
    record = Record("Counter", (Field("count", 0, 7), Field("total", 0, 28)))
    definition = RecordMachine(record, prefix_length=2, max_context=32, lower=-1, upper=28)

    def transition(step):
        count = step.state["count"]
        return Transition(
            dict(count=(count - 1).clip(0, 7), total=(step.state["total"] + count).clip(0, 28)),
            halt=count.equals(0),
        )

    model = compile_machine(definition.lower_machine(transition), max_context=definition.max_context)
    prefix = definition.initialize(count=3, total=0)
    trace = model.generate(prefix, definition.token_budget(3))
    for address, state in definition.records(trace, include_initial=True):
        print(address, state)
    assert trace == [3, 0, 2, 3, 1, 5, 0, 6, -1]


if __name__ == "__main__":
    main()
