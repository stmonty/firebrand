"""Reusable bounded logical memory backed by an append-only write log.

This builds symbolic expressions, never executes reads on the host at inference.
A read starts one microstep, then checks one write per microstep. Applications
explicitly hold their control state while busy and consume the value when ready.
"""

from dataclasses import dataclass

from .ir import Expr, constant, integer, latest, read_at, select
from .records import Field, Record


@dataclass(frozen=True)
class MemoryUpdate:
    fields: dict[str, Expr]
    busy: Expr  # Previous state was searching: suppress normal application work.
    ready: Expr  # This search microstep resolves a read.
    value: Expr  # Only meaningful when ready is 1.


@dataclass(frozen=True)
class WriteLog:
    cells: int
    value_limit: int
    max_context: int
    prefix: str = ""

    def __post_init__(self):
        for value in (self.cells, self.value_limit, self.max_context):
            integer(value)
        if self.cells < 1 or self.value_limit < 0 or not 1 <= self.max_context <= 2**25:
            raise ValueError("invalid write-log limits")
        if not isinstance(self.prefix, str):
            raise TypeError("memory field prefix must be a string")

    def name(self, field):
        return self.prefix + field

    @property
    def control_fields(self):
        return tuple(
            Field(self.name(name), 0, bound)
            for name, bound in (
                ("head", self.max_context),
                ("searching", 1),
                ("target", self.cells - 1),
                ("cursor", self.max_context),
            )
        )

    @property
    def entry_fields(self):
        return tuple(
            Field(self.name(name), 0, bound)
            for name, bound in (
                ("write_address", self.cells - 1),
                ("write_value", self.value_limit),
                ("parent", self.max_context),
            )
        )

    @property
    def fields(self):
        return self.control_fields + self.entry_fields

    def initial_state(self):
        """No writes or pending read. Reserve a zero record at token address 0."""
        return {field.name: 0 for field in self.fields}

    def step(self, step, record: Record, *, read=0, write=0, address=0, value=0):
        """Build one memory microstep and expose its completion signals.

        While busy, requests are ignored. When idle, read takes priority over
        write if both are asserted. Requests must be Boolean expressions.
        Address and value are explicitly saturated to this memory's bounds.
        A zero-filled record at physical address 0 is required for eager reads.
        The caller merges `fields` into its Transition and handles busy/ready.
        """
        declared = {field.name: field for field in record.fields}
        for field in self.fields:
            if declared.get(field.name) != field:
                raise ValueError(f"missing or incompatible memory field: {field.name}")
        state = lambda name: step.state[self.name(name)]
        busy = state("searching").clip(0, 1)
        cursor = state("cursor")
        key = record.read(cursor, self.name("write_address"))
        stored = record.read(cursor, self.name("write_value"))
        parent = record.read(cursor, self.name("parent"))
        found = select(cursor.equals(0), 1, (key - state("target")).equals(0))
        address = constant(address).clip(0, self.cells - 1)
        value = constant(value).clip(0, self.value_limit)
        store = select(busy, 0, select(read, 0, write))
        fields = dict(
            head=select(store, step.address, state("head")),
            searching=select(busy, 1 - found, read),
            target=select(busy, state("target"), address),
            cursor=select(busy, select(found, 0, parent), state("head")),
            write_address=address,
            write_value=value,
            parent=state("head"),
        )
        return MemoryUpdate(
            {self.name(name): expr for name, expr in fields.items()},
            busy,
            select(busy, found, 0),
            select(cursor.equals(0), 0, stored),
        )


@dataclass(frozen=True)
class IndexedMemory:
    """Logical memory whose reads resolve in the requesting microstep.

    A drop-in alternative to WriteLog for SequenceMachine. Each record carries
    at most one write in its final two fields: the value, then a key of
    address + 1 (0 means no write). One `latest` attention head finds the newest
    record whose key equals address + 1, so a read costs one head instead of one
    generated record per earlier write. Because the key is the last field, a
    record's own write is never visible while that record is being emitted:
    every field of a record sees the same memory. There are no pointer fields,
    so the token alphabet does not grow with the context length.
    """

    cells: int
    value_limit: int
    max_context: int
    prefix: str = ""

    def __post_init__(self):
        for value in (self.cells, self.value_limit, self.max_context):
            integer(value)
        if self.cells < 1 or self.value_limit < 0 or not 1 <= self.max_context <= 2**25:
            raise ValueError("invalid indexed-memory limits")
        if not isinstance(self.prefix, str):
            raise TypeError("memory field prefix must be a string")

    def name(self, field):
        return self.prefix + field

    @property
    def fields(self):
        return (Field(self.name("write_value"), 0, self.value_limit), Field(self.name("write_key"), 0, self.cells))

    def initial_state(self):
        return {field.name: 0 for field in self.fields}

    def step(self, step, record: Record, *, read=0, write=0, address=0, value=0):
        """Build one memory microstep; reads are ready immediately.

        When both are asserted, read takes priority over write, as in WriteLog.
        Address and value are saturated to this memory's bounds.
        """
        if record.fields[-2:] != self.fields:
            raise ValueError("indexed memory fields must be the final two record fields")
        width = record.width
        address = constant(address).clip(0, self.cells - 1)
        value = constant(value).clip(0, self.value_limit)
        key = address + 1
        # Tagged positions are write keys, starting with the initial record's.
        found_at = latest(key, period=width, phase=width - 1, start=step.prefix_length - width)
        found = (read_at(found_at) - key).equals(0)
        stored = read_at(found_at - 1).clip(0, self.value_limit)
        store = select(read, 0, write)
        fields = dict(write_value=select(store, value, 0), write_key=select(store, key, 0))
        return MemoryUpdate(
            {self.name(name): expr for name, expr in fields.items()},
            constant(0),
            constant(read),
            select(found, stored, 0),
        )
