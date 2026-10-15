"""Drafts compiled from a machine's source predict the transformer's tokens."""

import random

import pytest
from test_rv32 import PROGRAM

from firebrand import Machine, compile_machine, history, read_at, reference
from firebrand.draft import Draft
from firebrand.ir import match
from firebrand.machines.gcd import gcd_machine
from firebrand.machines.instruction import COUNTDOWN, InstructionSpec
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var
from firebrand.machines.pixel import DOWN, LEFT, RIGHT, PixelMachineSpec
from firebrand.machines.rv32 import RV32Spec
from firebrand.machines.scheduled_histogram import ScheduledHistogramSpec
from firebrand.machines.stack import POP, PUSH, StackSpec


def rv32_prefix(spec):
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(PROGRAM):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    return spec.encode(image, 0)


def cases():
    stack, lam, instruction, pixel = StackSpec(), LambdaSpec(max_context=256), InstructionSpec(), PixelMachineSpec()
    gcd = gcd_machine()
    histogram = ScheduledHistogramSpec(capacity=5, max_context=1024, indexed=True)
    rv32 = RV32Spec(ram_bytes=1280, max_context=1 << 13)
    identity = Lam(Var(0))
    return {
        "stack": (stack.machine(), stack.encode([(PUSH, 3), (PUSH, 5), (POP, 0)]), 13),
        "lambda": (
            lam.machine(),
            lam.serialize(App(App(identity, identity), Lam(Lam(Var(1)))))[0],
            lam.token_budget(16),
        ),
        "gcd": (gcd.lower_machine(), gcd.initialize(a=12, b=8, result=0), gcd.protocol.token_budget(90)),
        "instruction": (instruction.machine(), instruction.encode(COUNTDOWN), instruction.protocol.token_budget(70)),
        "pixel": (pixel.machine(), pixel.encode([RIGHT, DOWN, LEFT]), pixel.token_budget(3)),
        "histogram (latest heads)": (
            histogram.machine(),
            histogram.encode([1, 0, 1, 2, 1]),
            histogram.protocol.token_budget(60),
        ),
        "rv32 (match heads)": (rv32.machine(), rv32_prefix(rv32), rv32.protocol.token_budget(60 + 3 * 65)),
    }


CASES = cases()


@pytest.mark.parametrize("name", CASES)
def test_draft_equals_reference_and_transformer(name):
    machine, prefix, steps = CASES[name]
    model = compile_machine(machine)
    expected = model.generate(prefix, steps, backend="c")
    draft = Draft(machine)
    assert draft.run(prefix, steps) == expected and draft.error is None
    if len(reference.nodes(machine.next_value)) < 1000:  # The Python reference is slow on RV32IM.
        assert reference.run(machine, prefix, steps) == expected


def test_match_semantics_on_random_prompts():
    lookup = match(
        [read_at(0), read_at(1)], [history(1), history(0)], period=3, phase=2, start=5, sentinel=5, digit_limit=3
    )
    machine = Machine(read_at(lookup - 2), 0, 3, min_prefix=6)
    draft, rng = Draft(machine), random.Random(0)
    for _ in range(200):
        prompt = [rng.randint(0, 3) for _ in range(3 + 3 * rng.randint(1, 6) + rng.randint(0, 2))]
        assert draft.run(prompt, 8) == reference.run(machine, prompt, 8)


def test_a_failing_source_program_stops_the_draft():
    machine = Machine(read_at(history() + 2), 0, 9)  # Reads at the last token's value plus two.
    draft = Draft(machine)
    assert draft.run([0, 0, 9, 0], 5) == [0, 0, 9, 0, 9]  # read_at(2) gives 9; then read_at(11) fails.
    assert draft.error == "address outside causal prefix"
    with pytest.raises(ValueError, match="address outside causal prefix"):
        reference.run(machine, [0, 0, 9, 0], 5)


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", CASES)
def test_speculative_decoding_with_a_compiled_draft(name):
    from firebrand import speculative
    from firebrand.gpu import Verifier

    machine, prefix, steps = CASES[name]
    model = compile_machine(machine)
    expected = model.generate(prefix, steps, backend="c")
    verifier = Verifier(model)
    assert speculative.generate(model, prefix, Draft(machine), steps, verifier=verifier) == expected

    class Unreliable:
        """Right except for one token in fifty: each round ends at a wrong token."""

        def __init__(self):
            self.inner, self.rounds = Draft(machine), 0

        def run(self, tokens, steps):
            self.rounds += 1
            out = self.inner.run(tokens, steps)
            for i in range(len(tokens) + 49, len(out), 50):
                out[i] = machine.lower if out[i] != machine.lower else machine.lower + 1
            return out

    unreliable = Unreliable()
    assert speculative.generate(model, prefix, unreliable, steps, verifier=verifier) == expected
