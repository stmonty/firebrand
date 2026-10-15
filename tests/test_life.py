"""Conway's Game of Life in the weights, against an independent NumPy Life."""

import numpy as np
import pytest

from firebrand import compile_machine
from firebrand.draft import Draft
from firebrand.machines.life import GLIDER, LifeSpec, place


@pytest.fixture(scope="module")
def life():
    spec = LifeSpec(width=7, height=5, generations=10)
    return spec, compile_machine(spec.machine())


def reference(spec, grid):
    grids = [np.asarray(grid)]
    for _ in range(spec.generations):
        grids.append(spec.step(grids[-1]))
    return np.stack(grids)


@pytest.mark.parametrize("seed", range(8))
def test_random_soups_match_numpy(life, seed):
    spec, model = life
    grid = np.random.default_rng(seed).integers(0, 2, (spec.height, spec.width))
    trace = model.generate(spec.encode(grid), spec.max_context - spec.cells, backend="c")
    assert (spec.grids(trace) == reference(spec, grid)).all()


def test_a_glider_crosses_the_wrapping_edges(life):
    spec, model = life
    grid = place(spec, GLIDER, x=5, y=3)  # Partly across the right and bottom edges.
    trace = model.generate(spec.encode(grid), spec.max_context - spec.cells)
    grids = spec.grids(trace)
    assert (grids == reference(spec, grid)).all()
    assert all(g.sum() == 5 for g in grids)  # Still a glider every generation.


def test_one_model_runs_every_pattern_and_the_draft_agrees(life):
    spec, model = life
    draft = Draft(spec.machine())
    for seed in range(4):
        grid = np.random.default_rng(100 + seed).integers(0, 2, (spec.height, spec.width))
        prompt = spec.encode(grid)
        assert draft.run(prompt, spec.max_context - spec.cells) == model.generate(
            prompt, spec.max_context - spec.cells, backend="c"
        )
