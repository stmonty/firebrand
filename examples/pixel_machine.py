"""Run compiled movement and raster output; print only completed frames."""

from firebrand import compile_machine
from firebrand.machines.pixel import DOWN, LEFT, RIGHT, PixelMachineSpec


def main():
    spec = PixelMachineSpec()
    model = compile_machine(spec.machine())
    actions = [RIGHT, DOWN, LEFT]
    prefix = spec.encode(actions)
    trace = model.generate(prefix, spec.token_budget(len(actions)))
    for number, frame in enumerate(spec.frames(trace)):
        print(f"Frame {number}")
        print("\n".join("".join(".#"[color] for color in row) for row in frame))
    print(f"{len(trace) - len(prefix)} generated tokens, {len(spec.protocol.records(trace))} microsteps")


if __name__ == "__main__":
    main()
