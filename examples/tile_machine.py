"""The same weights execute movement on two different input maps."""

from firebrand import compile_machine
from firebrand.machines.pixel import DOWN, LEFT, RIGHT
from firebrand.machines.tile import TileMachineSpec


def main():
    spec = TileMachineSpec()
    model = compile_machine(spec.machine())
    actions = [RIGHT, DOWN, LEFT]
    for tiles in ([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0] * 4 for _ in range(3)]):
        prefix = spec.encode(actions, tiles)
        trace = model.generate(prefix, spec.token_budget(len(actions)))
        print("Map:", tiles)
        for frame in spec.frames(trace):
            print("\n".join("".join(".#@"[color] for color in row) for row in frame))
            print()
        print("Generated tokens:", len(trace) - len(prefix))


if __name__ == "__main__":
    main()
