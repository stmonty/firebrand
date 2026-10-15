"""Follow linked predecessor addresses using compiled attention queries."""

from firebrand import Machine, Transformer, compile_machine, history, read_at
from firebrand.reference import run


def main():
    # Positions 0..5 hold predecessor addresses. Position 6 supplies root 5.
    # The chain is 5 -> 4 -> 3 -> 2 -> 1 -> 0. We designate 0 as a halt value.
    prefix = [0, 0, 1, 2, 3, 4, 5]
    for name, expression in (
        ("one_link", read_at(history())),
        ("two_links", read_at(read_at(history()))),
    ):
        machine = Machine(expression, 0, 7, (0,))
        model = compile_machine(machine, max_context=64)
        path = f"artifacts/pointers_{name}"
        model.save(path)
        actual = Transformer.load(path).generate(prefix, 10)
        assert actual == run(machine, prefix, 10)
        print(f"{name}: emitted {actual[len(prefix) :]}")
        print("  schedule:", model.config["schedule"])


if __name__ == "__main__":
    main()
