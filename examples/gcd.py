"""Compile once, then follow data-dependent paths for different GCD inputs."""

from math import gcd

from firebrand import compile_machine
from firebrand.machines.gcd import gcd_machine


def main():
    program = gcd_machine()
    model = compile_machine(program.lower_machine())
    for a, b in ((12, 8), (7, 3), (0, 9), (0, 0)):
        trace = model.generate(program.initialize(a=a, b=b, result=0), program.protocol.token_budget(90))
        assert trace[-1] == -1
        records = program.protocol.records(trace)
        result = records[-1][1]["result"]
        assert result == gcd(a, b)
        print(f"gcd({a}, {b}) = {result}; {len(records)} microsteps")


if __name__ == "__main__":
    main()
