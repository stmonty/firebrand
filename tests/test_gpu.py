"""The GPU verifier predicts exactly the compiled decoder's tokens, and
speculative decoding with any draft returns the transformer's own trace."""

import random

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from test_torch_export import CASES

from firebrand import speculative
from firebrand.gpu import Verifier


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.__name__)
def case(request):
    return request.param()


@pytest.mark.parametrize(
    "options",
    [dict(chunk=97), dict(chunk=97, share=False), dict(chunk=97, graphs=False), dict()],
    ids=["chunks", "no-sharing", "no-graphs", "one-chunk"],
)
def test_predictions_equal_generated_tokens(case, options):
    model, prefix, trace = case
    predicted = Verifier(model, **options).predict(trace, len(prefix)).tolist()
    assert predicted[:-1] == trace[len(prefix) :]


def test_check_finds_the_first_wrong_token(case):
    model, prefix, trace = case
    verifier = Verifier(model, chunk=97)
    assert verifier.check(trace, len(prefix))[0] is None
    rng = random.Random(0)
    for _ in range(5):
        at = rng.randrange(len(prefix), len(trace))
        wrong = list(trace)
        wrong[at] = next(t for t in (0, 1, 2, -1) if t != trace[at])
        assert verifier.check(wrong, len(prefix)) == (at, trace[at])


def test_speculative_decoding_returns_the_transformers_trace(case):
    model, prefix, trace = case
    steps = len(trace) - len(prefix)
    verifier = Verifier(model, chunk=97)
    rng = random.Random(1)
    drafts = [
        trace[len(prefix) :],
        trace[len(prefix) : len(prefix) + steps // 2],
        [0] * steps,
        [rng.choice((0, 1, 2)) for _ in range(steps)],
    ]
    for draft in drafts:
        assert speculative.generate(model, prefix, draft, steps, verifier=verifier) == trace
