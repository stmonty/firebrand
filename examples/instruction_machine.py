"""A loop that updates logical memory, executed by transformer weights."""

from firebrand import Transformer, compile_machine
from firebrand.machines.instruction import COUNTDOWN, InstructionSpec, reference_run


def main():
    spec = InstructionSpec()
    model = compile_machine(spec.machine())
    model.save("artifacts/instruction_machine")
    model = Transformer.load("artifacts/instruction_machine")
    trace = model.generate(spec.encode(COUNTDOWN), spec.protocol.token_budget(40))
    print("Output:", spec.outputs(trace))
    print("Microsteps:", len(spec.records(trace)))
    print("Halted:", trace[-1] == -1)
    assert trace[-1] == -1
    assert spec.outputs(trace) == reference_run(COUNTDOWN)[1] == [3, 2, 1, 0]


if __name__ == "__main__":
    main()
