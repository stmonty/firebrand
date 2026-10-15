"""Count input categories with reusable memory, without an instruction interpreter."""

from dataclasses import dataclass

from ..ir import integer, read_at, select
from ..machine import RecordMachine, Transition
from ..memory import WriteLog
from ..records import Field, Record


@dataclass(frozen=True)
class HistogramSpec:
    cells: int = 3
    capacity: int = 8
    max_context: int = 1024

    def __post_init__(self):
        for value in (self.cells, self.capacity, self.max_context):
            integer(value)
        if self.cells < 1 or self.capacity < 1:
            raise ValueError("histogram limits must be positive")
        _ = self.protocol  # Validates the alphabet and context.

    @property
    def memory(self):
        return WriteLog(self.cells, self.capacity, self.max_context, prefix="mem_")

    @property
    def state(self):
        return Record(
            "Histogram",
            (
                Field("index", 0, self.capacity),
                Field("length", 0, self.capacity),
                Field("mode", 0, 2),
                Field("count", 0, self.capacity),
                *self.memory.fields,
            ),
        )

    @property
    def protocol(self):
        width = self.state.width
        prefix = ((width + self.capacity + width - 1) // width + 1) * width
        return RecordMachine(
            self.state, prefix, self.max_context, -1, max(self.max_context, self.capacity, self.cells, 2)
        )

    def encode(self, categories):
        categories = list(categories)
        if len(categories) > self.capacity:
            raise ValueError("too many categories")
        if any(not 0 <= integer(c) < self.cells for c in categories):
            raise ValueError("category outside histogram range")
        # Zero sentinel plus padded input, then the initial state.
        payload = [0] * self.state.width + categories + [0] * (self.capacity - len(categories))
        return self.protocol.initialize(
            payload, index=0, length=len(categories), mode=0, count=0, **self.memory.initial_state()
        )

    def machine(self):
        return self.protocol.lower_machine(self.transition)

    def transition(self, step):
        s = step.state
        category = read_at(self.state.width + s["index"].clip(0, self.capacity - 1))
        start, write = s["mode"].equals(0), s["mode"].equals(2)
        incremented = (s["count"] + 1).clip(0, self.capacity)
        memory = self.memory.step(step, self.state, read=start, write=write, address=category, value=incremented)
        # Application sequencing is explicit: start read, await result, write.
        return Transition(
            dict(
                index=select(
                    memory.busy, s["index"], select(write, (s["index"] + 1).clip(0, self.capacity), s["index"])
                ),
                length=s["length"],
                mode=select(memory.busy, select(memory.ready, 2, 1), select(write, 0, 1)),
                count=select(
                    memory.busy, select(memory.ready, memory.value, s["count"]), select(write, incremented, s["count"])
                ),
                **memory.fields,
            ),
            halt=select(memory.busy, 0, (s["index"] - s["length"]).equals(0)),
        )

    def result(self, trace):
        """Decode latest stored counts only after an observed halt."""
        records = self.protocol.records(trace)
        if len(trace) <= self.protocol.prefix_length or trace[-1] != self.protocol.halt_token:
            raise ValueError("histogram result requires HALT")
        counts = [0] * self.cells
        for base, state in records:
            if state["mem_head"] == base:
                counts[state["mem_write_address"]] = state["mem_write_value"]
        return counts
