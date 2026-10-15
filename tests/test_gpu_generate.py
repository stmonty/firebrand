"""Serial generation on the GPU produces the compiled decoder's tokens."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import test_program as programs
from test_torch_export import rv32_case

from firebrand.gpu_generate import GPUGenerator


@pytest.mark.parametrize(
    "compiled, args",
    [
        (programs.gcd, (60, 48)),
        (programs.histogram, ([1, 3, 3, 0, 4, 4, 4, 2, 1, 0, 0, 3, 4, 2, 2, 1],)),
        (programs.sieve, (30,)),
    ],
    ids=["gcd", "histogram", "sieve"],
)
def test_programs(compiled, args):
    prompt = compiled.prompt(*args)
    steps = 3000
    assert GPUGenerator(compiled.model).generate(prompt, steps) == compiled.model.generate(prompt, steps, backend="c")


def test_rv32_interpreter():
    model, prefix, trace = rv32_case()
    assert GPUGenerator(model).generate(prefix, len(trace) - len(prefix) + 5) == trace
