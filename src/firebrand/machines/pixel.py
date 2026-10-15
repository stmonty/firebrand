"""A tiny simulation and renderer: replayed actions move a dot; every pixel is emitted."""

from dataclasses import dataclass

from ..ir import integer, read_at, select
from ..machine import RecordMachine, Transition
from ..records import Field, Record
from .framebuffer import FRAME_END, NO_OUTPUT, PIXEL, Framebuffer

STAY, LEFT, RIGHT, UP, DOWN = range(5)


@dataclass(frozen=True)
class PixelMachineSpec:
    width: int = 4
    height: int = 3
    capacity: int = 4
    max_context: int = 1024

    def __post_init__(self):
        _ = self.framebuffer, self.protocol  # Built eagerly: they validate the limits.

    @property
    def framebuffer(self):
        return Framebuffer(self.width, self.height, 2, self.capacity)

    @property
    def state(self):
        return Record(
            "PixelMachine",
            (
                Field("player_x", 0, self.width - 1),
                Field("player_y", 0, self.height - 1),
                Field("input_index", 0, self.capacity),
                Field("input_length", 0, self.capacity),
                Field("mode", 0, 2),  # 0: consume input, 1: pixels, 2: frame end.
                Field("scan_x", 0, self.width - 1),
                Field("scan_y", 0, self.height - 1),
                *self.framebuffer.fields,
            ),
        )

    @property
    def payload_capacity(self):
        return self.capacity

    @property
    def protocol(self):
        width = self.state.width
        prefix = ((self.payload_capacity + width - 1) // width + 1) * width
        upper = max(DOWN, *(field.upper for field in self.state.fields))
        return RecordMachine(self.state, prefix, self.max_context, -1, upper)

    def encode(self, actions, *, x=0, y=0):
        actions = list(actions)
        if len(actions) > self.capacity:
            raise ValueError("too many input actions")
        if any(not STAY <= integer(action) <= DOWN for action in actions):
            raise ValueError("unknown input action")
        return self.protocol.initialize(
            actions,
            player_x=x,
            player_y=y,
            input_index=0,
            input_length=len(actions),
            mode=0,
            scan_x=0,
            scan_y=0,
            output_kind=NO_OUTPUT,
            output_frame=0,
            output_index=0,
            output_color=0,
        )

    def token_budget(self, actions):
        if not 0 <= integer(actions) <= self.capacity:
            raise ValueError("input count outside capacity")
        return self.protocol.token_budget(actions * (self.width * self.height + 2))

    def machine(self):
        return self.protocol.lower_machine(self.transition)

    def movement(self, state, action):
        return (
            (state["player_x"] + action.equals(RIGHT) - action.equals(LEFT)).clip(0, self.width - 1),
            (state["player_y"] + action.equals(DOWN) - action.equals(UP)).clip(0, self.height - 1),
        )

    def pixel(self, state):
        return select(
            (state["scan_x"] - state["player_x"]).equals(0), (state["scan_y"] - state["player_y"]).equals(0), 0
        )

    def transition(self, step):
        s = step.state
        move, draw, finish = (s["mode"].equals(i) for i in range(3))
        # Input is fixed prefix data, and the padded slot is valid even on HALT.
        action = read_at(s["input_index"].clip(0, self.capacity - 1))
        next_x, next_y = self.movement(s, action)
        row_end = s["scan_x"].equals(self.width - 1)
        frame_end = select(row_end, s["scan_y"].equals(self.height - 1), 0)
        color = self.pixel(s)
        return Transition(
            dict(
                player_x=select(move, next_x, s["player_x"]),
                player_y=select(move, next_y, s["player_y"]),
                input_index=select(finish, (s["input_index"] + 1).clip(0, self.capacity), s["input_index"]),
                input_length=s["input_length"],
                mode=select(move, 1, select(draw, select(frame_end, 2, 1), 0)),
                scan_x=select(draw, select(row_end, 0, (s["scan_x"] + 1).clip(0, self.width - 1)), 0),
                scan_y=select(draw, select(row_end, (s["scan_y"] + 1).clip(0, self.height - 1), s["scan_y"]), 0),
                output_kind=select(draw, PIXEL, select(finish, FRAME_END, NO_OUTPUT)),
                output_frame=s["input_index"].clip(0, self.capacity - 1),
                output_index=select(draw, s["scan_y"] * self.width + s["scan_x"], 0).clip(
                    0, self.width * self.height - 1
                ),
                output_color=select(draw, color, 0),
            ),
            halt=select(move, (s["input_index"] - s["input_length"]).equals(0), 0),
        )

    def frames(self, trace, *, require_halt=True):
        """Mechanically replay events, without consulting player state or inputs."""
        records = self.protocol.records(trace, allow_partial=not require_halt)
        halted = len(trace) > self.protocol.prefix_length and trace[-1] == self.protocol.halt_token
        if require_halt and not halted:
            raise ValueError("complete run requires HALT")
        return self.framebuffer.replay(self.framebuffer.events(records), allow_partial=not halted)
