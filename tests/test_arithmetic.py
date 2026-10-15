"""The compiled arithmetic stream decides exactly what direct parsing decides."""

import random

import pytest

from firebrand import compile_machine
from firebrand.machines.arithmetic import EQUALS, OTHER, PASS, PLUS, SPACE, TIMES, ArithmeticSpec, reference_next


@pytest.fixture(scope="module")
def stream():
    spec = ArithmeticSpec(digits=6, max_context=256)
    return spec, compile_machine(spec.machine())


def digits(number):
    return [int(c) for c in str(number)]


def test_answers_are_exact(stream):
    _spec, model = stream
    rng = random.Random(0)
    for _ in range(60):
        a, b = rng.randrange(10 ** rng.randint(1, 6)), rng.randrange(10 ** rng.randint(1, 6))
        op = rng.choice((TIMES, PLUS))
        prompt = [OTHER, SPACE] + digits(a) + [op] + [SPACE] * rng.randint(0, 1) + digits(b) + [EQUALS]
        out = []
        for token in model.generate(prompt, 20, backend="c")[len(prompt) :]:
            if token == PASS:
                break
            out.append(token)
        assert out == [SPACE] + digits(a * b if op == TIMES else a + b)


def test_every_decision_matches_direct_parsing_on_messy_text(stream):
    """Random token soups with expression fragments: wrong order, no operator,
    operands that are too long, extra spaces, text after an answer."""
    torch = pytest.importorskip("torch")
    from firebrand.torch_export import to_torch

    spec, model = stream
    module = to_torch(model)
    rng = random.Random(1)
    pieces = [[OTHER], [SPACE], [EQUALS], [TIMES], [PLUS], [SPACE, EQUALS]]
    for _ in range(150):
        codes = [OTHER]
        while len(codes) < rng.randint(4, 40):
            if rng.random() < 0.5:
                codes += digits(rng.randrange(10 ** rng.randint(1, 8)))
            else:
                codes += rng.choice(pieces)
        # One parallel pass of the exported weights gives the decision after every prefix.
        decisions = module(torch.tensor(codes)).argmax(dim=1).tolist()
        assert decisions == [reference_next(codes[:end], spec.digits) for end in range(1, len(codes) + 1)], codes


def test_longest_operands_and_carries(stream):
    _spec, model = stream
    for a, b, op in [(999999, 999999, TIMES), (999999, 1, PLUS), (0, 123456, TIMES), (1, 0, PLUS)]:
        prompt = [OTHER] + digits(a) + [op] + digits(b) + [EQUALS]
        out = model.generate(prompt, 20, backend="c")[len(prompt) :]
        expected = digits(a * b if op == TIMES else a + b)
        assert out[: len(expected) + 2] == [SPACE] + expected + [PASS]


def test_language_model_tokens_map_to_codes():
    from firebrand.machines.arithmetic import code_of

    assert [code_of(t) for t in ["7", " *", "×", " \\", "times", " ", " =", "+", "²", "12", " the"]] == [
        7,
        TIMES,
        TIMES,
        SPACE,
        TIMES,
        SPACE,
        EQUALS,
        PLUS,
        OTHER,
        OTHER,
        OTHER,
    ]


def test_the_merged_model_forces_exact_answers_then_hands_back():
    """A stand-in language model that always prefers the word "blah": the merged
    model writes the exact product after "=", then the language model resumes."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from firebrand.bridge import ExactArithmeticLM

    vocabulary = [str(d) for d in range(10)] + [" ", " *", " =", "Q", " blah", "<eos>"]

    class Tokenizer:
        eos_token_id = vocabulary.index("<eos>")

        def __len__(self):
            return len(vocabulary)

        def decode(self, ids, skip_special_tokens=False):
            return "".join(vocabulary[i] for i in ids)

        def tokenize(self, text):
            return [text]

        def convert_tokens_to_ids(self, tokens):
            return (
                [vocabulary.index(t) for t in tokens]
                if isinstance(tokens, list)
                else vocabulary.index(tokens)
                if tokens in vocabulary
                else 0
            )

    class Blah(torch.nn.Module):
        def forward(self, input_ids, past_key_values=None, use_cache=True):
            logits = torch.zeros(1, input_ids.shape[1], len(vocabulary))
            logits[..., vocabulary.index(" blah")] = 1
            return SimpleNamespace(logits=logits, past_key_values=None)

    lm = ExactArithmeticLM(Blah(), Tokenizer(), digits=4, max_context=128)
    prompt = [vocabulary.index(t) for t in ["Q", "1", "2", " *", " ", "3", "4", " ="]]
    new, forced = lm.generate(torch.tensor(prompt), 6)
    assert lm.tokenizer.decode(new.tolist()) == " 408 blah blah"
    assert forced == 4
