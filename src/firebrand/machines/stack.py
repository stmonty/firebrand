"""Bounded stack test bed: command records in the prompt, frames in the trace.

One instruction emits four tokens: [value, parent, top, next_pc]. A push's record
is also its allocated frame; a pop preserves old frames and changes top. Pop on
an empty stack returns zero. Opcode 0 halts by emitting the reserved token -1.
"""

from dataclasses import dataclass

from ..ir import select
from ..machine import RecordMachine, Transition
from ..records import Field, Record

HALT, PUSH, POP = 0, 1, 2


@dataclass(frozen=True)
class StackSpec:
    capacity: int = 8  # Command slots, including the final halt instruction.
    max_context: int = 128
    value_limit: int = 7

    def __post_init__(self):
        if any(type(v) is not int for v in (self.capacity, self.max_context, self.value_limit)):
            raise TypeError("stack limits must be integers")
        if self.capacity < 2 or self.capacity % 2:
            raise ValueError("capacity must be a positive even number of command slots")
        if not 0 <= self.value_limit <= self.max_context:
            raise ValueError("invalid value limit")
        if self.max_context < self.prefix_length + 4 * (self.capacity - 1):
            raise ValueError("context must fit the longest program and its final halt query")

    @property
    def prefix_length(self):
        return 8 + 2 * self.capacity

    @property
    def command(self):
        return Record("Command", (Field("opcode", 0, 2), Field("argument", 0, self.value_limit)))

    @property
    def frame(self):
        return Record(
            "Frame",
            (
                Field("value", 0, self.value_limit),
                Field("parent", 0, self.max_context),
                Field("top", 0, self.max_context),
                Field("next_pc", 0, self.max_context),
            ),
        )

    @property
    def protocol(self):
        return RecordMachine(self.frame, self.prefix_length, self.max_context, -1, self.max_context)

    def encode(self, program):
        """Serialize commands, sentinel, and initial state; never evaluate them."""
        if len(program) >= self.capacity:
            raise ValueError("program must leave a command slot for halt")
        tokens = self.frame.pack(value=0, parent=0, top=0, next_pc=0)
        for opcode, argument in program:
            if opcode not in (PUSH, POP):
                raise ValueError("program instructions must be PUSH or POP")
            if opcode == POP and argument != 0:
                raise ValueError("POP has no argument; use zero")
            tokens.extend(self.command.pack(opcode=opcode, argument=argument))
        for _ in range(self.capacity - len(program)):
            tokens.extend(self.command.pack(opcode=HALT, argument=0))
        return self.protocol.initialize(tokens, value=0, parent=0, top=0, next_pc=4)

    def machine(self):
        return self.protocol.lower_machine(self.transition)

    def transition(self, step):
        top = step.state["top"]
        pc = step.state["next_pc"]
        opcode = self.command.read(pc, "opcode")
        argument = self.command.read(pc, "argument")
        old_value = self.frame.read(top, "value")
        old_parent = self.frame.read(top, "parent")
        is_push = opcode.equals(PUSH)
        value = select(is_push, argument, old_value)
        new_top = select(is_push, step.address, old_parent)
        next_pc = (pc + self.command.width).clip(4, 4 + 2 * (self.capacity - 1))
        # All branches are eagerly evaluated. Sentinel frame at address zero
        # makes empty-stack reads valid, even when a branch is not selected.
        return Transition(
            dict(value=value, parent=top, top=new_top, next_pc=next_pc),
            halt=opcode.equals(HALT),
        )

    def decode(self, trace):
        """Read completed output records for display/checking, not execution."""
        return [record for _, record in self.protocol.records(trace)]


def reference_stack(program):
    """Independent conventional list-based semantics for results after each op."""
    stack, results = [], []
    for opcode, argument in program:
        if opcode == PUSH:
            stack.append(argument)
            value = argument
        elif opcode == POP:
            value = stack.pop() if stack else 0
        else:
            raise ValueError("unknown stack operation")
        results.append((value, stack.copy()))
    return results
