"""Run with: python examples/demo.py (after installing the package)."""

from firebrand import Machine, Transformer, compile_machine, history
from firebrand.reference import run


def main():
    cases = {
        "countdown": (Machine((history() - 1).relu(), 0, 31, (0,)), [10]),
        "oscillator": (Machine(-history(), -3, 3), [3]),
        "fibonacci": (Machine((history() + history(1)).clip(0, 31), 0, 31, (31,)), [0, 1]),
    }
    for name, (machine, prefix) in cases.items():
        model = compile_machine(machine)
        path = f"artifacts/{name}"
        model.save(path)
        restored = Transformer.load(path)
        actual = restored.generate(prefix, max_steps=12)
        assert actual == run(machine, prefix, max_steps=12)
        print(f"{name}: {actual}")
        print(f"  {model.config['heads']} attention heads, {model.config['layers']} FFN blocks; saved {path}")


if __name__ == "__main__":
    main()
