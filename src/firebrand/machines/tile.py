"""Runtime map data used by compiled collision and top-down rendering rules."""

from dataclasses import dataclass

from ..ir import integer, read_at, select
from .framebuffer import Framebuffer
from .pixel import PixelMachineSpec


@dataclass(frozen=True)
class TileMachineSpec(PixelMachineSpec):
    """A binary wall map follows the fixed-capacity action table in the prefix.

    Colors are 0=floor, 1=wall, 2=player. Dimensions and capacity are compiled;
    map cells, starting position, and actions are data. A blocked move leaves
    the position unchanged and still produces a frame. Screen edges clip moves.
    """

    @property
    def framebuffer(self):
        return Framebuffer(self.width, self.height, 3, self.capacity)

    @property
    def map_address(self):
        return self.capacity

    @property
    def payload_capacity(self):
        return self.capacity + self.width * self.height

    def encode(self, actions, tiles, *, x=0, y=0):
        rows = [list(row) for row in tiles]
        if len(rows) != self.height or any(len(row) != self.width for row in rows):
            raise ValueError("map dimensions must match the compiled screen")
        flat = [integer(cell) for row in rows for cell in row]
        if any(cell not in (0, 1) for cell in flat):
            raise ValueError("map cells must be zero (floor) or one (wall)")
        prefix = super().encode(actions, x=x, y=y)
        if rows[y][x]:
            raise ValueError("player must start on floor")
        prefix[self.map_address : self.map_address + len(flat)] = flat
        return prefix

    def tile(self, x, y):
        # All expressions run eagerly, including outside their selected phase.
        # Clamp coordinates independently so every read stays in the map region.
        address = self.map_address + y.clip(0, self.height - 1) * self.width + x.clip(0, self.width - 1)
        return read_at(address).clip(0, 1)

    def movement(self, state, action):
        x, y = super().movement(state, action)
        wall = self.tile(x, y)
        return select(wall, state["player_x"], x), select(wall, state["player_y"], y)

    def pixel(self, state):
        player = super().pixel(state)
        return select(player, 2, self.tile(state["scan_x"], state["scan_y"]))
