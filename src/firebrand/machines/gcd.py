"""Euclid's subtraction algorithm using named branches and backward jumps."""

from ..ir import integer, select
from ..memory import WriteLog
from ..records import Field, Record
from ..sequence import Assign, Branch, Halt, Jump, Label, SequenceMachine


def gcd_machine(value_limit=15, max_context=1024):
    """Build a bounded GCD program; memory is unused by this application.

    Inputs are nonnegative, including gcd(0, 0) = 0. Use initialize(a=..., b=...,
    result=0), then lower_machine() and the normal compiler/decoder.
    """
    if integer(value_limit) < 0:
        raise ValueError("value limit must be nonnegative")
    state = Record("GCD", tuple(Field(name, 0, value_limit) for name in ("a", "b", "result")))
    return SequenceMachine(
        state,
        WriteLog(1, value_limit, max_context, prefix="mem_"),
        (
            Label("check"),
            Branch(lambda s: s["a"].equals(0), "finish", "check_b"),
            Label("check_b"),
            Branch(lambda s: s["b"].equals(0), "finish", "compare"),
            Label("compare"),
            Branch(lambda s: (s["a"] - s["b"]).equals(0), "finish", "choose"),
            Label("choose"),
            Branch(lambda s: (s["a"] - s["b"]).relu().equals(0), "reduce_b", "reduce_a"),
            Label("reduce_a"),
            Assign({"a": lambda s: (s["a"] - s["b"]).clip(0, value_limit)}),
            Jump("check"),
            Label("reduce_b"),
            Assign({"b": lambda s: (s["b"] - s["a"]).clip(0, value_limit)}),
            Jump("check"),
            Label("finish"),
            Assign({"result": lambda s: select(s["a"].equals(0), s["b"], s["a"])}),
            Halt(),
        ),
    )
