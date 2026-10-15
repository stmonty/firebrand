"""The histogram expressed as logical operations with generated continuations."""

from dataclasses import dataclass

from ..ir import integer, read_at
from ..memory import IndexedMemory
from ..records import Field, Record
from ..sequence import Assign, Read, SequenceMachine, Write
from .histogram import HistogramSpec


@dataclass(frozen=True)
class ScheduledHistogramSpec(HistogramSpec):
    indexed: bool = False  # IndexedMemory instead of the linked WriteLog.

    @property
    def memory(self):
        if self.indexed:
            return IndexedMemory(self.cells, self.capacity, self.max_context, prefix="mem_")
        return super().memory

    @property
    def sequence(self):
        state = Record(
            "Histogram",
            (Field("index", 0, self.capacity), Field("length", 0, self.capacity), Field("count", 0, self.capacity)),
        )
        width = state.width + 1 + len(self.memory.fields)

        def category(s):
            return read_at(width + s["index"].clip(0, self.capacity - 1))

        return SequenceMachine(
            state,
            self.memory,
            (
                Read("count", category),
                Assign({"count": lambda s: (s["count"] + 1).clip(0, self.capacity)}),
                Write(category, lambda s: s["count"]),
                Assign({"index": lambda s: (s["index"] + 1).clip(0, self.capacity)}),
            ),
            halt=lambda s: (s["index"] - s["length"]).equals(0),
            payload_capacity=self.capacity,
        )

    @property
    def state(self):
        return self.sequence.record

    @property
    def protocol(self):
        return self.sequence.protocol

    def encode(self, categories):
        # Match the manual histogram input contract.
        categories = list(categories)
        if len(categories) > self.capacity:
            raise ValueError("too many categories")
        if any(not 0 <= integer(c) < self.cells for c in categories):
            raise ValueError("category outside histogram range")
        return self.sequence.initialize(categories, index=0, length=len(categories), count=0)

    def machine(self):
        return self.sequence.lower_machine()

    def result(self, trace):
        if not self.indexed:
            return super().result(trace)
        records = self.protocol.records(trace)
        if len(trace) <= self.protocol.prefix_length or trace[-1] != self.protocol.halt_token:
            raise ValueError("histogram result requires HALT")
        counts = [0] * self.cells
        for _, state in records:
            if state["mem_write_key"]:
                counts[state["mem_write_key"] - 1] = state["mem_write_value"]
        return counts
