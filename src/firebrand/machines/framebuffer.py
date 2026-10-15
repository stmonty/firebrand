"""A mechanical output adapter for explicit raster pixels and frame boundaries.

This module knows no game state, geometry, visibility, or movement rules.
Frame numbers begin at zero; each frame supplies every pixel in raster order.
"""

from dataclasses import dataclass

from ..ir import integer
from ..records import Field

NO_OUTPUT, PIXEL, FRAME_END = 0, 1, 2


@dataclass(frozen=True)
class FrameEvent:
    kind: int
    frame: int
    index: int = 0
    color: int = 0


@dataclass(frozen=True)
class Framebuffer:
    width: int
    height: int
    colors: int
    frame_capacity: int

    def __post_init__(self):
        if any(integer(v) < 1 for v in (self.width, self.height, self.colors, self.frame_capacity)):
            raise ValueError("framebuffer dimensions, colors, and capacity must be positive")

    @property
    def fields(self):
        return (
            Field("output_kind", 0, FRAME_END),
            Field("output_frame", 0, self.frame_capacity - 1),
            Field("output_index", 0, self.width * self.height - 1),
            Field("output_color", 0, self.colors - 1),
        )

    def events(self, records):
        """Extract events from complete decoded records; ignore NO_OUTPUT records."""
        for _, state in records:
            kind = integer(state["output_kind"])
            if kind != NO_OUTPUT:
                yield FrameEvent(kind, state["output_frame"], state["output_index"], state["output_color"])

    def replay(self, events, *, allow_partial=False):
        """Return completed frames as immutable rows of palette indices.

        A full set of pixels without FRAME_END is still incomplete. If partial
        output is allowed, validate but omit its unfinished last frame. Duplicate,
        missing, out-of-order, and out-of-range pixels/frames are rejected.
        """
        frames, pixels = [], []
        for event in events:
            kind, frame, index, color = (integer(v) for v in (event.kind, event.frame, event.index, event.color))
            if frame != len(frames) or not 0 <= frame < self.frame_capacity:
                raise ValueError("unexpected frame number")
            if kind == PIXEL:
                if index != len(pixels) or not 0 <= index < self.width * self.height:
                    raise ValueError("pixels must cover the frame in raster order")
                if not 0 <= color < self.colors:
                    raise ValueError("pixel color outside palette")
                pixels.append(color)
            elif kind == FRAME_END:
                if index != 0 or color != 0:
                    raise ValueError("frame end must have zero index and color")
                if len(pixels) != self.width * self.height:
                    raise ValueError("frame end before all pixels arrived")
                frames.append(tuple(tuple(pixels[y * self.width : (y + 1) * self.width]) for y in range(self.height)))
                pixels = []
            else:
                raise ValueError("unknown framebuffer event")
        if pixels and not allow_partial:
            raise ValueError("incomplete frame: FRAME_END required")
        return frames
