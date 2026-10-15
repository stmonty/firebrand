import itertools

import pytest

from firebrand import Transformer, compile_machine
from firebrand.machines.framebuffer import FRAME_END
from firebrand.machines.pixel import DOWN, LEFT, RIGHT, STAY, UP
from firebrand.machines.tile import TileMachineSpec


def reference(tiles, actions, x, y):
    height, width = len(tiles), len(tiles[0])
    frames, positions = [], []
    for action in actions:
        dx, dy = {STAY: (0, 0), LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1)}[action]
        nx, ny = min(width - 1, max(0, x + dx)), min(height - 1, max(0, y + dy))
        if tiles[ny][nx] == 0:
            x, y = nx, ny
        frame = [list(row) for row in tiles]
        frame[y][x] = 2
        frames.append(tuple(map(tuple, frame)))
        positions.append((x, y))
    return frames, positions


def test_every_small_map_floor_start_and_action_with_one_model():
    spec = TileMachineSpec(width=2, height=2, capacity=1, max_context=128)
    model = compile_machine(spec.machine())
    for cells in itertools.product((0, 1), repeat=4):
        tiles = [cells[:2], cells[2:]]
        for y, x in itertools.product(range(2), repeat=2):
            if tiles[y][x]:
                continue
            for action in range(5):
                prefix = spec.encode([action], tiles, x=x, y=y)
                trace = model.generate(prefix, spec.token_budget(1))
                expected, positions = reference(tiles, [action], x, y)
                assert trace[-1] == -1 and spec.frames(trace) == expected
                final = spec.protocol.records(trace)[-1][1]
                assert (final["player_x"], final["player_y"]) == positions[-1]


def test_sequence_reload_and_display_uses_only_events(tmp_path, monkeypatch):
    spec = TileMachineSpec()
    model = compile_machine(spec.machine())
    model.save(tmp_path)
    tiles = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]
    actions = [RIGHT, DOWN, RIGHT, UP]
    prefix = spec.encode(actions, tiles)

    def forbidden(*args):
        raise AssertionError("host game logic called during generation")

    monkeypatch.setattr(TileMachineSpec, "movement", forbidden)
    monkeypatch.setattr(TileMachineSpec, "pixel", forbidden)
    trace = Transformer.load(tmp_path).generate(prefix, spec.token_budget(4))
    assert model.generate(prefix, spec.token_budget(4), use_cache=False) == trace
    expected, positions = reference(tiles, actions, 0, 0)
    assert spec.frames(trace) == expected
    records = spec.protocol.records(trace)
    assert [(s["player_x"], s["player_y"]) for _, s in records if s["output_kind"] == FRAME_END] == positions
    assert spec.framebuffer.replay(spec.framebuffer.events(records)) == expected
    # Partial execution cannot publish a frame without its complete marker.
    partial = trace[: len(prefix) + 13 * spec.state.width]
    assert spec.frames(partial, require_halt=False) == []


def test_single_cell_empty_input_and_map_layout():
    spec = TileMachineSpec(width=1, height=1, capacity=2, max_context=128)
    model = compile_machine(spec.machine())
    for actions in ([], [RIGHT], [LEFT, DOWN]):
        prefix = spec.encode(actions, [[0]])
        assert prefix[spec.map_address] == 0
        trace = model.generate(prefix, spec.token_budget(len(actions)))
        assert spec.frames(trace) == [((2,),)] * len(actions)
    # The map starts after action capacity, not after the actual action count.
    other = TileMachineSpec()
    tiles = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    prefix = other.encode([STAY], tiles)
    assert prefix[: other.capacity] == [0] * other.capacity
    assert prefix[other.map_address : other.map_address + 12] == sum(tiles, [])


@pytest.mark.parametrize(
    "tiles", [[], [[0]], [[0] * 4] * 2, [[0] * 3] * 3, [[0, 2, 0, 0]] + [[0] * 4] * 2, [[0, -1, 0, 0]] + [[0] * 4] * 2]
)
def test_invalid_maps(tiles):
    with pytest.raises(ValueError):
        TileMachineSpec().encode([STAY], tiles)


def test_invalid_start_and_noninteger_cells():
    spec = TileMachineSpec()
    with pytest.raises(ValueError, match="floor"):
        spec.encode([STAY], [[1] * 4] * 3)
    with pytest.raises(ValueError):
        spec.encode([], [[0] * 4] * 3, x=-1)
    with pytest.raises(TypeError):
        spec.encode([], [[False] * 4] * 3)
