"""Bounded accumulator machine with a persistent write log (LOAD walks it newest first)."""

from dataclasses import dataclass

from ..ir import integer, read_at, select
from ..machine import RecordMachine, Transition
from ..memory import WriteLog
from ..records import Field, Record

HALT, SET, ADD, SUB, LOAD, STORE, JZ, JMP, OUT = range(9)
NAMES = ("HALT", "SET", "ADD", "SUB", "LOAD", "STORE", "JZ", "JMP", "OUT")


@dataclass(frozen=True)
class InstructionSpec:
    capacity: int = 16  # Includes an automatically appended HALT.
    memory_cells: int = 4
    value_limit: int = 31
    max_context: int = 1024

    def __post_init__(self):
        for v in (self.capacity, self.memory_cells, self.value_limit, self.max_context):
            integer(v)
        if self.capacity < 2 or self.memory_cells < 1 or self.value_limit < 0:
            raise ValueError("invalid instruction machine limits")
        _ = self.protocol  # Validates the alphabet and context.

    @property
    def upper(self):
        return max(self.max_context, self.capacity, self.memory_cells, self.value_limit, OUT)

    @property
    def memory(self):
        return WriteLog(self.memory_cells, self.value_limit, self.max_context)

    @property
    def state(self):
        return Record(
            "InstructionState",
            (
                Field("pc", 0, self.capacity - 1),
                Field("acc", 0, self.value_limit),
                *self.memory.control_fields,
                Field("output", 0, self.value_limit),
                Field("output_valid", 0, 1),
                *self.memory.entry_fields,
            ),
        )

    @property
    def prefix_length(self):
        width = self.state.width
        payload = width + 2 * self.capacity
        return ((payload + width - 1) // width + 1) * width

    @property
    def protocol(self):
        return RecordMachine(self.state, self.prefix_length, self.max_context, -1, self.upper)

    def encode(self, program):
        """Validate and serialize instructions, without executing the program.

        Arithmetic saturates to 0..value_limit; memory starts at zero. Jump
        arguments are instruction indices, and may target the appended HALT.
        """
        program = list(program)
        if len(program) >= self.capacity:
            raise ValueError("program must leave a slot for HALT")
        payload = [0] * self.state.width  # Valid null log record for eager reads.
        for opcode, argument in program:
            integer(opcode)
            integer(argument)
            if not HALT <= opcode <= OUT:
                raise ValueError("unknown opcode")
            limit = (
                self.memory_cells - 1
                if opcode in (LOAD, STORE)
                else len(program)
                if opcode in (JZ, JMP)
                else 0
                if opcode in (HALT, OUT)
                else self.value_limit
            )
            if not 0 <= argument <= limit:
                raise ValueError(f"invalid argument for {NAMES[opcode]}")
            payload.extend((opcode, argument))
        payload.extend([0, 0] * (self.capacity - len(program)))
        return self.protocol.initialize(payload, **{f.name: 0 for f in self.state.fields})

    def machine(self):
        return self.protocol.lower_machine(self.transition)

    def transition(self, step):
        s = step.state
        command = self.state.width + 2 * s["pc"]
        opcode, arg = read_at(command), read_at(command + 1)
        is_op = {op: opcode.equals(op) for op in range(9)}
        advance = (s["pc"] + 1).clip(0, self.capacity - 1)
        jump = arg.clip(0, self.capacity - 1)
        address = arg.clip(0, self.memory_cells - 1)
        immediate = arg.clip(0, self.value_limit)
        memory = self.memory.step(
            step, self.state, read=is_op[LOAD], write=is_op[STORE], address=address, value=s["acc"]
        )
        acc = select(
            is_op[SET],
            immediate,
            select(
                is_op[ADD],
                (s["acc"] + immediate).clip(0, self.value_limit),
                select(is_op[SUB], (s["acc"] - immediate).clip(0, self.value_limit), s["acc"]),
            ),
        )
        pc = select(
            is_op[JMP],
            jump,
            select(is_op[JZ], select(s["acc"].equals(0), jump, advance), select(is_op[LOAD], s["pc"], advance)),
        )
        return Transition(
            dict(
                pc=select(memory.busy, select(memory.ready, advance, s["pc"]), pc),
                acc=select(memory.busy, select(memory.ready, memory.value, s["acc"]), acc),
                **memory.fields,
                output=s["acc"],
                output_valid=select(memory.busy, 0, is_op[OUT]),
            ),
            halt=select(memory.busy, 0, is_op[HALT]),
        )

    def records(self, trace):
        return self.protocol.records(trace, allow_partial=True)

    def outputs(self, trace):
        """Completed OUT events, possibly a prefix if execution has not halted."""
        return [r["output"] for _, r in self.records(trace) if r["output_valid"]]


def reference_run(program, *, value_limit=31, memory_cells=4, max_steps=100):
    """Independent instruction interpreter using a mutable Python memory array.

    Its budget counts instructions, whereas compiled budgets count microsteps.
    Inputs are expected to satisfy InstructionSpec.encode's program contract.
    """
    memory, pc, acc, outputs, states = [0] * memory_cells, 0, 0, [], []
    program = list(program) + [(HALT, 0)]
    for _ in range(max_steps):
        opcode, arg = program[pc]
        if opcode == HALT:
            return states, outputs, True
        next_pc = pc + 1
        if opcode == SET:
            acc = arg
        elif opcode == ADD:
            acc = min(value_limit, acc + arg)
        elif opcode == SUB:
            acc = max(0, acc - arg)
        elif opcode == LOAD:
            acc = memory[arg]
        elif opcode == STORE:
            memory[arg] = acc
        elif opcode == JZ:
            next_pc = arg if acc == 0 else next_pc
        elif opcode == JMP:
            next_pc = arg
        elif opcode == OUT:
            outputs.append(acc)
        else:
            raise ValueError("unknown opcode")
        pc = next_pc
        states.append((pc, acc, memory.copy(), outputs.copy()))
    return states, outputs, False


# Count down a repeatedly overwritten memory cell, emitting 3, 2, 1, 0.
COUNTDOWN = [(SET, 3), (STORE, 0), (LOAD, 0), (OUT, 0), (JZ, 8), (SUB, 1), (STORE, 0), (JMP, 2)]
