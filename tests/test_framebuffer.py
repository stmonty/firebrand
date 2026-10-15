import itertools

import pytest

from firebrand import Transformer, compile_machine
from firebrand.machines.framebuffer import FRAME_END, PIXEL, Framebuffer, FrameEvent
from firebrand.machines.pixel import DOWN, LEFT, RIGHT, STAY, UP, PixelMachineSpec


def reference(width, height, actions, x, y):
    """Conventional mutable state and array rendering, independent of the IR."""
    frames, positions = [], []
    for action in actions:
        if action == LEFT:
            x = max(0, x - 1)
        elif action == RIGHT:
            x = min(width - 1, x + 1)
        elif action == UP:
            y = max(0, y - 1)
        elif action == DOWN:
            y = min(height - 1, y + 1)
        image = [[0] * width for _ in range(height)]
        image[y][x] = 1
        frames.append(tuple(map(tuple, image)))
        positions.append((x, y))
    return frames, positions


def test_all_two_action_replays_and_start_positions_match_every_pixel():
    spec = PixelMachineSpec(width=2, height=2, capacity=2, max_context=256)
    model = compile_machine(spec.machine())
    for x, y in itertools.product(range(2), repeat=2):
        for actions in itertools.product(range(5), repeat=2):
            trace = model.generate(spec.encode(actions, x=x, y=y), spec.token_budget(2))
            expected, positions = reference(2, 2, actions, x, y)
            assert trace[-1] == -1
            assert spec.frames(trace) == expected
            records = spec.protocol.records(trace)
            assert [(s["player_x"], s["player_y"]) for _, s in records if s["output_kind"] == FRAME_END] == positions
            assert len(records) == 12


def test_non_square_frames_saved_weights_and_no_host_transition(tmp_path, monkeypatch):
    spec = PixelMachineSpec()
    model = compile_machine(spec.machine())
    model.save(tmp_path)

    def forbidden(*args):
        raise AssertionError("host transition executed during inference")

    monkeypatch.setattr(PixelMachineSpec, "transition", forbidden)
    actions = [RIGHT, DOWN, LEFT]
    prefix = spec.encode(actions)
    trace = Transformer.load(tmp_path).generate(prefix, spec.token_budget(3))
    assert model.generate(prefix, spec.token_budget(3), use_cache=False) == trace
    assert spec.frames(trace) == reference(4, 3, actions, 0, 0)[0]
    # Display receives only event fields, with no player position or inputs.
    events = list(spec.framebuffer.events(spec.protocol.records(trace)))
    assert spec.framebuffer.replay(events) == spec.frames(trace)
    assert len(events) == 3 * (4 * 3 + 1)


def test_empty_input_and_single_pixel_screen():
    spec = PixelMachineSpec(width=1, height=1, capacity=1, max_context=64)
    model = compile_machine(spec.machine())
    prefix = spec.encode([])
    trace = model.generate(prefix, spec.token_budget(0))
    assert trace == prefix + [-1] and spec.frames(trace) == []
    for action in (STAY, LEFT, RIGHT, UP, DOWN):
        trace = model.generate(spec.encode([action]), spec.token_budget(1))
        assert spec.frames(trace) == [((1,),)]


def test_frame_boundary_is_required_even_after_every_pixel():
    spec = PixelMachineSpec(width=2, height=1, capacity=2, max_context=128)
    model = compile_machine(spec.machine())
    prefix = spec.encode([RIGHT, LEFT])
    # First frame is complete after four records. The next has all its pixels
    # after seven, but has not emitted its frame-end record yet.
    trace = model.generate(prefix, 7 * spec.state.width)
    expected = reference(2, 1, [RIGHT], 0, 0)[0]
    assert spec.frames(trace, require_halt=False) == expected
    with pytest.raises(ValueError, match="HALT"):
        spec.frames(trace)
    # One token of the next FRAME_END record cannot publish a frame.
    trace = model.generate(prefix, 7 * spec.state.width + 1)
    assert spec.frames(trace, require_halt=False) == expected
    # An apparent HALT after an unfinished frame is a malformed output stream.
    incomplete = model.generate(prefix, 7 * spec.state.width) + [-1]
    with pytest.raises(ValueError, match="incomplete frame"):
        spec.frames(incomplete)


@pytest.mark.parametrize(
    "events",
    [
        [FrameEvent(PIXEL, 1, 0, 0)],
        [FrameEvent(PIXEL, 0, 1, 0)],
        [FrameEvent(PIXEL, 0, 0, 2)],
        [FrameEvent(PIXEL, 0, 0, 0), FrameEvent(PIXEL, 0, 0, 1)],
        [FrameEvent(FRAME_END, 0)],
        [FrameEvent(PIXEL, 0, 0, 0), FrameEvent(PIXEL, 0, 1, 1), FrameEvent(FRAME_END, 0, 1)],
        [FrameEvent(3, 0)],
    ],
)
def test_adapter_rejects_malformed_output(events):
    with pytest.raises(ValueError):
        Framebuffer(2, 1, 2, 1).replay(events, allow_partial=True)


def test_no_frame_is_inferred_from_complete_pixels_without_marker():
    display = Framebuffer(1, 1, 2, 1)
    pixels = [FrameEvent(PIXEL, 0, 0, 1)]
    with pytest.raises(ValueError, match="FRAME_END"):
        display.replay(pixels)
    assert display.replay(pixels, allow_partial=True) == []
    assert display.replay(pixels + [FrameEvent(FRAME_END, 0)]) == [((1,),)]


def test_input_and_configuration_contracts():
    spec = PixelMachineSpec(capacity=1)
    for actions in ([5], [-1], [0, 0]):
        with pytest.raises(ValueError):
            spec.encode(actions)
    with pytest.raises(TypeError):
        spec.encode([True])
    with pytest.raises(ValueError):
        spec.encode([STAY], x=spec.width)
    for config in (dict(width=0), dict(height=-1), dict(capacity=0), dict(max_context=1)):
        with pytest.raises(ValueError):
            PixelMachineSpec(**config)
    with pytest.raises(ValueError):
        spec.token_budget(2)
    with pytest.raises(ValueError, match="budget"):
        PixelMachineSpec(max_context=32).token_budget(1)
