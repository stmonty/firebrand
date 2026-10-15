"""Allocate persistent stack frames through transformer-generated tokens."""

from firebrand import Transformer, compile_machine
from firebrand.machines.stack import POP, PUSH, StackSpec, reference_stack


def main():
    spec = StackSpec()
    program = [(PUSH, 2), (PUSH, 5), (POP, 0), (PUSH, 7), (POP, 0), (POP, 0), (POP, 0)]
    model = compile_machine(spec.machine(), max_context=spec.max_context)
    model.save("artifacts/stack")
    trace = Transformer.load("artifacts/stack").generate(spec.encode(program), 4 * len(program) + 1)
    records = spec.decode(trace)
    for operation, record, (expected, stack) in zip(program, records, reference_stack(program)):
        assert record["value"] == expected
        print(f"{operation}: value={record['value']}, top={record['top']}; stack={stack}")
    assert trace[-1] == -1
    print(f"Allocated trace: {len(trace)} tokens; {model.config['width']} residual coordinates")


if __name__ == "__main__":
    main()
