"""Match heads: multi-digit keys computed at earlier positions, with a sentinel."""

import random

import numpy as np
import pytest

from firebrand import Machine, Transformer, compile_machine, history, read_at
from firebrand.ir import match
from firebrand.reference import evaluate

# Prompt: two query digits, a padding token, then 3-token records [value, hi, lo].
# Keys are each record's last two tokens; the first record (ending at 5) is the
# sentinel, so a lookup with no match returns that record's value.
LOOKUP = match(
    [read_at(0), read_at(1)], [history(1), history(0)], period=3, phase=2, start=5, sentinel=5, digit_limit=3
)
VALUE = read_at(LOOKUP - 2)


@pytest.fixture(scope="module")
def model():
    return compile_machine(Machine(VALUE, 0, 3, min_prefix=6), max_context=64)


def prompts(count, seed=0):
    rng = random.Random(seed)
    for _ in range(count):
        records = rng.randint(1, 6)
        yield [rng.randint(0, 3) for _ in range(3 + 3 * records + rng.randint(0, 2))]


def test_reference_semantics():
    prompt = [2, 1, 0, 9, 9, 9, 1, 2, 1, 3, 0, 0, 2, 2, 1]
    assert evaluate(LOOKUP, prompt) == 14  # Newest record with key (2, 1).
    assert evaluate(LOOKUP, prompt[:12]) == 8  # Only the older one.
    assert evaluate(LOOKUP, [0, 3] + prompt[2:12]) == 5  # No match: the sentinel.
    assert evaluate(LOOKUP, [0, 0] + prompt[2:9]) == 5  # Sentinel keys count as zero.
    assert evaluate(LOOKUP, [0, 0] + prompt[2:12]) == 11  # ...but a newer (0, 0) key wins.


def test_every_decoder_matches_the_semantics(model):
    for prompt in prompts(300):
        expected = evaluate(VALUE, prompt)
        assert int(np.argmax(model.logits(prompt))) == expected  # Scanning reference.
        assert model.generate(prompt, 1)[-1] == expected  # Cached dictionary.
        assert model.generate(prompt, 1, use_cache=False)[-1] == expected
        assert model.generate(prompt, 1, backend="c")[-1] == expected  # Compiled decoder.


def test_generated_tokens_become_keys(model):
    # Each generated token can itself end a record whose key later lookups see.
    prompt = [1, 1, 0, 0, 0, 0]
    trace = model.generate(prompt, 12)
    assert trace == model.generate(prompt, 12, use_cache=False) == model.generate(prompt, 12, backend="c")
    for n in range(len(prompt), len(trace)):
        assert trace[n] == evaluate(VALUE, trace[:n])


def test_contracts(model, tmp_path):
    with pytest.raises(ValueError, match="sentinel"):
        match([read_at(0)], [history()], period=3, phase=2, start=5, sentinel=4)
    with pytest.raises(ValueError, match="digits"):
        match([read_at(0)], [history(), history(1)], period=3, phase=2, start=5, sentinel=5)
    # No key matches 3 and the sentinel (position 9) is not in the prompt yet.
    early = compile_machine(
        Machine(read_at(match([read_at(0)], [history()], period=1, phase=0, start=1, sentinel=9)), 0, 3), max_context=16
    )
    for options in ({}, {"use_cache": False}, {"backend": "c"}):
        with pytest.raises(ValueError, match="sentinel"):
            early.generate([3, 1, 2], 1, **options)
    assert model.config["head_kinds"].count("match") == 1
    model.save(tmp_path)
    prompt = next(prompts(1, seed=5))
    assert Transformer.load(tmp_path).generate(prompt, 4) == model.generate(prompt, 4)


def test_position_digits_and_quotients_in_every_decoder():
    from firebrand import position_digit, position_quotient

    expr = (position_digit(0, 200) + position_digit(1, 200) + position_quotient(7, -3)).clip(0, 300)
    model = compile_machine(Machine(expr, 0, 300), max_context=400)
    for length in (1, 3, 55, 56, 57, 300):
        prompt = [0] * length
        expected = evaluate(expr, prompt)
        for options in ({}, {"use_cache": False}, {"backend": "c"}):
            assert model.generate(prompt, 1, **options)[-1] == expected
    assert evaluate(position_quotient(7, -3), [0]) == -1
    assert evaluate(position_digit(1, 200), [0] * 57) == 1  # 256 >> 8


def test_tag_end_hides_later_keys():
    # Same records as LOOKUP, but keys ending at or after position 11 are invisible.
    hidden = match(
        [read_at(0), read_at(1)],
        [history(1), history(0)],
        period=3,
        phase=2,
        start=5,
        sentinel=5,
        digit_limit=3,
        end=11,
    )
    value = read_at(hidden - 2)
    ended = compile_machine(Machine(value, 0, 3, min_prefix=6), max_context=64)
    prompt = [2, 1, 0, 0, 0, 0, 1, 2, 1, 3, 0, 0, 2, 2, 1]
    assert evaluate(hidden, prompt) == 8  # Not 14: it ends after 11.
    for options in ({}, {"use_cache": False}, {"backend": "c"}):
        assert ended.generate(prompt, 1, **options)[-1] == prompt[6]
    with pytest.raises(ValueError, match="end"):
        match([read_at(0)], [history()], period=3, phase=2, start=5, sentinel=5, end=5)
