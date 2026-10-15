"""Conway's Game of Life on a torus.

Each token is a cell, row by row, one generation after another; the prompt is
the first generation. A cell reads its nine cells in the previous generation
(N = width × height tokens earlier) at addresses computed from its position, and
applies the rule. `LifeSpec.step` is an independent NumPy Life.
"""

from dataclasses import dataclass

import numpy as np

from ..ir import Machine, position, position_mod, read_at, select


@dataclass(frozen=True)
class LifeSpec:
    width: int = 32
    height: int = 32
    generations: int = 64  # Generations after the first, for the context size.

    @property
    def cells(self):
        return self.width * self.height

    @property
    def max_context(self):
        return self.cells * (self.generations + 1)

    def machine(self):
        width, height, n = self.width, self.height, self.cells
        # The cell being predicted is the next token: c = (position + 1) mod N.
        cell = position_mod(n, 1)
        x = cell % width
        y = cell.lookup(tuple(c // width for c in range(n)))
        # The previous generation's cell 0, and wrapped neighbour coordinates.
        before = position() + 1 - n - cell
        xs = {-1: select(x.equals(0), width - 1, x - 1), 0: x, 1: select(x.equals(width - 1), 0, x + 1)}
        ys = {-1: select(y.equals(0), height - 1, y - 1), 0: y, 1: select(y.equals(height - 1), 0, y + 1)}
        neighbours = [
            read_at(before + ys[dy] * width + xs[dx]) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
        ]
        count = neighbours[0]
        for neighbour in neighbours[1:]:
            count = count + neighbour
        alive = read_at(before + cell)
        rule = select(count.equals(3), 1, select(alive, count.equals(2), 0))
        return Machine(rule, 0, 1, min_prefix=n, context_limit=self.max_context)

    def encode(self, grid):
        grid = np.asarray(grid, dtype=np.int64)
        if grid.shape != (self.height, self.width) or not np.isin(grid, (0, 1)).all():
            raise ValueError(f"expected a {self.height}x{self.width} grid of 0s and 1s")
        return grid.reshape(-1).tolist()

    def grids(self, trace):
        """Complete generations in a trace, as [generations, height, width]."""
        whole = len(trace) // self.cells * self.cells
        return np.asarray(trace[:whole], dtype=np.int64).reshape(-1, self.height, self.width)

    @staticmethod
    def step(grid):
        """One generation of Life on a torus, by NumPy (the independent reference)."""
        count = sum(
            np.roll(np.roll(grid, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
        )
        return ((count == 3) | ((grid == 1) & (count == 2))).astype(np.int64)


def place(spec, pattern, x=0, y=0):
    """A spec-sized grid with a pattern (rows of '.' and 'O') at (x, y)."""
    grid = np.zeros((spec.height, spec.width), dtype=np.int64)
    for dy, row in enumerate(pattern.strip().splitlines()):
        for dx, char in enumerate(row.strip()):
            if char == "O":
                grid[(y + dy) % spec.height, (x + dx) % spec.width] = 1
    return grid


GLIDER = """
.O.
..O
OOO
"""

GOSPER_GLIDER_GUN = """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
"""
