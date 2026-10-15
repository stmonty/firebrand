"""Declare record state and a transition; generate the scalar-token protocol.

The transition callback runs once at construction time on symbolic expressions.
Inference needs only the resulting ordinary Machine weights, not this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .ir import Expr, Machine, integer, position, position_mod, select
from .records import Record


@dataclass(frozen=True)
class Transition:
    next_fields: Mapping[str, Expr | int]
    halt: Expr | int = 0


class _StateReads(Mapping):
    def __init__(self, record, previous_address):
        self._record = record
        self._previous_address = previous_address
        self._cache = {}

    def __getitem__(self, name):
        self._record.offset(name)  # Reject misspellings at construction time.
        if name not in self._cache:
            self._cache[name] = self._record.read(self._previous_address, name)
        return self._cache[name]

    def __iter__(self):
        return (field.name for field in self._record.fields)

    def __len__(self):
        return self._record.width


class Step:
    """Symbolic inputs: previous complete state and this step's allocation address."""

    def __init__(self, definition):
        next_position = position() + 1
        # A positional input feature, so record width does not cap the context.
        self._phase = position_mod(definition.record.width, 1)
        self.prefix_length = definition.prefix_length
        self.address = (next_position - self._phase).clip(0, definition.max_context)
        previous = (self.address - definition.record.width).clip(0, definition.max_context)
        self.state = _StateReads(definition.record, previous)


@dataclass(frozen=True)
class RecordMachine:
    record: Record
    prefix_length: int
    max_context: int
    lower: int
    upper: int
    halt_token: int = -1

    def __post_init__(self):
        for value in (self.prefix_length, self.max_context, self.lower, self.upper, self.halt_token):
            integer(value)
        if self.lower > self.upper or not self.lower <= self.halt_token <= self.upper:
            raise ValueError("invalid alphabet or halt token")
        if self.prefix_length < self.record.width or self.prefix_length % self.record.width:
            raise ValueError("prefix must include an initial record and align to its width")
        if not self.prefix_length <= self.max_context <= 2**25:
            raise ValueError("context must cover prefix and be at most 2**25")
        for field in self.record.fields:
            if field.lower < self.lower or field.upper > self.upper:
                raise ValueError("field range exceeds token alphabet")
            if field.lower <= self.halt_token <= field.upper:
                raise ValueError("halt token must be reserved outside every field range")

    def lower_machine(self, transition):
        """Call a symbolic transition builder once and lower its output protocol.

        All fields must be assigned explicitly. Halt is tested at the start of a
        record; all expressions remain eager even when halt is true.
        """
        step = Step(self)
        update = transition(step)
        if not isinstance(update, Transition):
            raise TypeError("transition builder must return Transition")
        values = dict(update.next_fields)
        if set(values) != {field.name for field in self.record.fields}:
            raise ValueError("transition must assign every record field exactly once")
        first = self.record.fields[0].name
        values[first] = select(update.halt, self.halt_token, values[first])
        emitted = self.record.emit(step._phase, **values)
        return Machine(
            emitted,
            self.lower,
            self.upper,
            (self.halt_token,),
            min_prefix=self.prefix_length,
            context_limit=self.max_context,
        )

    def initialize(self, payload=(), **state):
        """Pad an opaque syntax/program payload, then append checked initial state."""
        payload = list(payload)
        capacity = self.prefix_length - self.record.width
        if len(payload) > capacity:
            raise ValueError("payload does not fit before initial state")
        if any(not self.lower <= integer(value) <= self.upper or value == self.halt_token for value in payload):
            raise ValueError("payload contains a token outside its data alphabet")
        padding = (
            0
            if self.lower <= 0 <= self.upper and self.halt_token != 0
            else next((v for v in (self.lower, self.lower + 1) if v != self.halt_token and v <= self.upper), None)
        )
        if padding is None:
            raise ValueError("alphabet has no data token for padding")
        return payload + [padding] * (capacity - len(payload)) + self.record.pack(**state)

    def token_budget(self, steps):
        """Complete records plus a single token to observe a final halt."""
        if integer(steps) < 0 or self.prefix_length + self.record.width * steps > self.max_context:
            raise ValueError("step budget exceeds compiled context or is negative")
        return self.record.width * steps + 1

    def records(self, trace, *, include_initial=False, allow_partial=False):
        """Decode complete records; optionally ignore a budget-truncated tail."""
        if len(trace) < self.prefix_length:
            raise ValueError("trace is shorter than initial prefix")
        tail = trace[self.prefix_length :]
        halted = bool(tail) and tail[-1] == self.halt_token
        if halted:
            tail = tail[:-1]
        if self.halt_token in tail:
            raise ValueError("halt token inside a record or before the end of the trace")
        partial = len(tail) % self.record.width
        if partial and (halted or not allow_partial):
            raise ValueError("trace ends in an incomplete record")
        end = self.prefix_length + len(tail) - partial
        start = self.prefix_length - self.record.width if include_initial else self.prefix_length
        return [(base, self.record.unpack(trace, base)) for base in range(start, end, self.record.width)]
