"""Lower logical operations and named control flow into record microsteps.

Callbacks build expressions once during lowering. Inference executes only the
resulting weights. This is an explicit operation DSL, not a Python compiler.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .ir import constant, integer, select
from .machine import RecordMachine, Transition
from .memory import IndexedMemory, WriteLog
from .records import Field, Record


@dataclass(frozen=True)
class Read:
    destination: str
    address: object  # Integer/expression, or callback from application state.


@dataclass(frozen=True)
class Write:
    address: object
    value: object


@dataclass(frozen=True)
class Assign:
    updates: Mapping  # Simultaneous assignments from the state before this op.

    def __post_init__(self):
        object.__setattr__(self, "updates", MappingProxyType(dict(self.updates)))


@dataclass(frozen=True)
class Label:
    name: str  # Compile-time marker; consumes no microstep.


@dataclass(frozen=True)
class Jump:
    target: str


@dataclass(frozen=True)
class Branch:
    condition: object
    when_true: str
    when_false: str


@dataclass(frozen=True)
class Halt:
    """Emit the reserved halt token when this operation is reached."""


def _expression(value, state):
    return constant(value(state) if callable(value) else value)


@dataclass(frozen=True)
class SequenceMachine:
    state: Record  # Application fields only; internal state is added below.
    memory: WriteLog | IndexedMemory
    operations: tuple
    halt: object = 0  # Checked at the beginning of each iteration only.
    payload_capacity: int = 0

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.executable:
            raise ValueError("sequence must contain at least one executable operation")
        labels = self.labels
        if integer(self.payload_capacity) < 0:
            raise ValueError("payload capacity must be nonnegative")
        names = {field.name for field in self.state.fields}
        internal = {"sequence_pc"} | {field.name for field in self.memory.fields}
        if names & internal:
            raise ValueError("application fields collide with generated sequence/memory fields")
        fields = {field.name: field for field in self.state.fields}
        for op in self.executable:
            targets = (
                (op.target,)
                if isinstance(op, Jump)
                else (op.when_true, op.when_false)
                if isinstance(op, Branch)
                else ()
            )
            for target in targets:
                if not isinstance(target, str) or target not in labels:
                    raise ValueError(f"unknown control-flow target: {target!r}")
            if isinstance(op, Read):
                if op.destination not in names:
                    raise ValueError("read destination must be an application field")
                field = fields[op.destination]
                if field.lower > 0 or field.upper < self.memory.value_limit:
                    raise ValueError("read destination must cover the memory value range")
            elif isinstance(op, Assign):
                if not op.updates or not set(op.updates) <= names:
                    raise ValueError("assignments must name existing application fields")
            elif not isinstance(op, (Write, Jump, Branch, Halt)):
                raise TypeError("unsupported sequence operation")
        _ = self.protocol  # Validates the alphabet and context.

    @property
    def executable(self):
        return tuple(op for op in self.operations if not isinstance(op, Label))

    @property
    def labels(self):
        labels, index = {}, 0
        for op in self.operations:
            if isinstance(op, Label):
                if not isinstance(op.name, str) or not op.name or op.name in labels:
                    raise ValueError("labels must have unique nonempty string names")
                labels[op.name] = index
            else:
                index += 1
        if any(target == index for target in labels.values()):
            raise ValueError("a label must precede an executable operation; add Halt at the end")
        return labels

    @property
    def record(self):
        return Record(
            self.state.name + "Sequence",
            self.state.fields + (Field("sequence_pc", 0, len(self.executable) - 1),) + self.memory.fields,
        )

    @property
    def payload_address(self):
        return self.record.width  # The first record is the zero memory sentinel.

    @property
    def protocol(self):
        width = self.record.width
        prefix = ((width + self.payload_capacity + width - 1) // width + 1) * width
        # The halt token is -1, or below every field if a field can hold -1.
        fields = self.record.fields
        halt = -1 if not any(f.lower <= -1 <= f.upper for f in fields) else min(f.lower for f in fields) - 1
        lower = min(halt, *(f.lower for f in fields))
        upper = max(0, *(f.upper for f in fields))
        return RecordMachine(self.record, prefix, self.memory.max_context, lower, upper, halt_token=halt)

    def initialize(self, payload=(), **state):
        payload = list(payload)
        if len(payload) > self.payload_capacity:
            raise ValueError("payload exceeds sequence capacity")
        # Check application keys before adding generated fields.
        self.state.pack(**state)
        return self.protocol.initialize(
            [0] * self.record.width + payload, **state, sequence_pc=0, **self.memory.initial_state()
        )

    def lower_machine(self):
        return self.protocol.lower_machine(self._transition)

    def _transition(self, step):
        # Restrict callbacks to application state, not generated implementation fields.
        state = MappingProxyType({field.name: step.state[field.name] for field in self.state.fields})
        pc = step.state["sequence_pc"]
        operations, labels = self.executable, self.labels
        active = [pc.equals(i) for i in range(len(operations))]
        # Exactly one operation is active, so each choice below is a sum of gated
        # terms (a balanced tree: depth grows with log of the program's length).
        reads, writes, addresses, values = [], [], [], []
        built = []
        for enabled, op in zip(active, operations):
            if isinstance(op, Read):
                reads.append(enabled)
                addresses.append(select(enabled, _expression(op.address, state), 0))
                built.append(None)
            elif isinstance(op, Write):
                writes.append(enabled)
                addresses.append(select(enabled, _expression(op.address, state), 0))
                values.append(select(enabled, _expression(op.value, state), 0))
                built.append(None)
            elif isinstance(op, Assign):
                built.append({name: _expression(expr, state) for name, expr in op.updates.items()})
            elif isinstance(op, Branch):
                built.append(_expression(op.condition, state))
            else:
                built.append(None)
        memory = self.memory.step(
            step, self.record, read=_flag(reads), write=_flag(writes), address=_sum(addresses), value=_sum(values)
        )
        fields = {field.name: field for field in self.record.fields}
        chosen = {name: [] for name in state}  # name -> [(complete, value)]
        targets, completions, halts = [], [], []
        for i, (enabled, op, assignments) in enumerate(zip(active, operations, built)):
            # A read advances only on completion; other ops take one idle microstep.
            complete = select(enabled, memory.ready if isinstance(op, Read) else 1 - memory.busy, 0)
            target = (i + 1) % len(operations)
            if isinstance(op, Jump):
                target = labels[op.target]
            elif isinstance(op, Branch):
                target = select(assignments, labels[op.when_true], labels[op.when_false])
            elif isinstance(op, Halt):
                halts.append(complete)
                target = i
            completions.append(complete)
            targets.append(select(complete, target, 0))
            if isinstance(op, Read):
                chosen[op.destination].append((complete, memory.value))
            elif isinstance(op, Assign):
                for name, expr in assignments.items():
                    chosen[name].append((complete, expr))
        updates = dict(state)
        for name, options in chosen.items():
            if options:
                field = fields[name]
                value = _sum([select(c, v, 0) for c, v in options]).clip(field.lower, field.upper)
                updates[name] = select(_flag([c for c, _ in options]), value, state[name])
        updates.update(memory.fields)
        updates["sequence_pc"] = select(_flag(completions), _sum(targets).clip(0, len(operations) - 1), pc)
        explicit_halt = _flag(halts)
        halt = select(active[0], select(memory.busy, 0, _expression(self.halt, state)), 0)
        return Transition(updates, halt=select(explicit_halt, 1, halt))


def _sum(terms):
    """A balanced sum of expressions (0 for none)."""
    terms = [constant(t) for t in terms]
    if not terms:
        return constant(0)
    while len(terms) > 1:
        terms = [terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i] for i in range(0, len(terms), 2)]
    return terms[0]


def _flag(flags):
    """1 if any of some mutually exclusive 0/1 flags is set."""
    return _sum(flags).clip(0, 1)
