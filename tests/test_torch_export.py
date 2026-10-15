"""The exported PyTorch module computes the same function as the compiled model.

It runs every position in parallel with dense causal attention, so one forward
pass over a generated trace checks every generated token (teacher forcing).
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_rv32 import PROGRAM

from firebrand import Machine, compile_machine, history, read_at
from firebrand.ir import match
from firebrand.machines.gcd import gcd_machine
from firebrand.machines.instruction import COUNTDOWN, InstructionSpec
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var
from firebrand.machines.pixel import DOWN, LEFT, RIGHT, PixelMachineSpec
from firebrand.machines.rv32 import RV32Spec
from firebrand.machines.scheduled_histogram import ScheduledHistogramSpec
from firebrand.machines.stack import POP, PUSH, StackSpec
from firebrand.torch_export import to_torch


def check(model, prefix, trace, **options):
    """Every generated token is the module's argmax given the tokens before it,
    and its final logits equal the NumPy reference's."""
    module = to_torch(model, **options)
    logits = module(torch.tensor(trace))
    predicted = (logits.argmax(dim=1) + model.config["lower"]).tolist()
    assert predicted[len(prefix) - 1 : len(trace) - 1] == trace[len(prefix) :]
    np.testing.assert_array_equal(logits[-1].numpy(), model.logits(trace))


def stack_case():
    spec = StackSpec()
    model = compile_machine(spec.machine())
    prefix = spec.encode([(PUSH, 3), (PUSH, 5), (POP, 0)])
    return model, prefix, model.generate(prefix, 13)


def lambda_case():
    spec = LambdaSpec(max_context=256)
    model = compile_machine(spec.machine())
    identity = Lam(Var(0))
    prefix = spec.serialize(App(App(identity, identity), Lam(Lam(Var(1)))))[0]
    return model, prefix, model.generate(prefix, spec.token_budget(16))


def gcd_case():
    program = gcd_machine()
    model = compile_machine(program.lower_machine())
    prefix = program.initialize(a=12, b=8, result=0)
    return model, prefix, model.generate(prefix, program.protocol.token_budget(90))


def instruction_case():
    spec = InstructionSpec()
    model = compile_machine(spec.machine())
    prefix = spec.encode(COUNTDOWN)
    return model, prefix, model.generate(prefix, spec.protocol.token_budget(70))


def pixel_case():
    spec = PixelMachineSpec()
    model = compile_machine(spec.machine())
    prefix = spec.encode([RIGHT, DOWN, LEFT])
    return model, prefix, model.generate(prefix, spec.token_budget(3))


def rv32_case():
    """The RV32IM interpreter: absolute and match heads, byte arithmetic, and
    microcoded division."""
    spec = RV32Spec(ram_bytes=1280, max_context=1 << 13)
    model = compile_machine(spec.machine())
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(PROGRAM):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    prefix = spec.encode(image, 0)
    trace = model.generate(prefix, spec.protocol.token_budget(60 + 3 * 65), backend="c")
    assert spec.halted(trace)
    return model, prefix, trace


def histogram_case():
    """Indexed memory: `latest` heads find the newest write to an address."""
    spec = ScheduledHistogramSpec(capacity=5, max_context=1024, indexed=True)
    model = compile_machine(spec.machine())
    prefix = spec.encode([1, 0, 1, 2, 1])
    trace = model.generate(prefix, spec.protocol.token_budget(60))
    assert spec.result(trace) == [1, 3, 1]
    return model, prefix, trace


CASES = [stack_case, lambda_case, gcd_case, instruction_case, pixel_case, histogram_case, rv32_case]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__)
@pytest.mark.parametrize("attention", ["hardmax", "softmax"])
def test_module_predicts_every_generated_token(case, attention):
    check(*case(), attention=attention)


def test_match_heads_with_random_prompts():
    lookup = match(
        [read_at(0), read_at(1)], [history(1), history(0)], period=3, phase=2, start=5, sentinel=5, digit_limit=3
    )
    model = compile_machine(Machine(read_at(lookup - 2), 0, 3, min_prefix=6), max_context=64)
    module = to_torch(model, attention="softmax")
    rng = random.Random(0)
    for _ in range(100):
        prompt = [rng.randint(0, 3) for _ in range(3 + 3 * rng.randint(1, 6) + rng.randint(0, 2))]
        expected = int(np.argmax(model.logits(prompt)))
        assert int(module(torch.tensor(prompt))[-1].argmax()) == expected


def test_module_generates_alone():
    model, prefix, trace = gcd_case()
    assert to_torch(model).generate(prefix, len(trace) - len(prefix)) == trace
