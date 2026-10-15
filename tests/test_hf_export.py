"""Compiled models exported as stock Phi-3 checkpoints generate the same tokens."""

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import test_program as programs
from test_torch_export import (
    gcd_case,
    histogram_case,
    instruction_case,
    lambda_case,
    pixel_case,
    stack_case,
)

from firebrand import Machine, compile_machine, history, read_at
from firebrand.hf_export import prompt_ids, to_huggingface, tokens_from_ids
from firebrand.ir import match
from firebrand.machines.life import GLIDER, LifeSpec, place


def generate(path, model, prompt, steps):
    hf = transformers.AutoModelForCausalLM.from_pretrained(path)
    out = hf.generate(torch.tensor([prompt_ids(model, prompt)]), max_new_tokens=steps, do_sample=False)
    return tokens_from_ids(model, out[0].tolist())


@pytest.mark.parametrize(
    "case",
    [stack_case, lambda_case, gcd_case, instruction_case, pixel_case, histogram_case],
    ids=lambda case: case.__name__,
)
def test_machines(case, tmp_path):
    model, prefix, trace = case()
    to_huggingface(model, tmp_path, max_context=len(trace) + 8)
    assert generate(tmp_path, model, prefix, len(trace) - len(prefix)) == trace


@pytest.mark.parametrize(
    "compiled, args",
    [
        (programs.gcd, (60, 48)),
        (programs.sieve, (12,)),
        (programs.first_over, ([0, 3, 0, 9, 12, 1, 0, 4, 20, 2, 0, 5], 8)),
    ],
    ids=["gcd", "sieve", "first_over"],
)
def test_programs(compiled, args, tmp_path):
    prompt = compiled.prompt(*args)
    trace = compiled.model.generate(prompt, 3000, backend="c")
    to_huggingface(compiled.model, tmp_path, max_context=len(trace) + 4)
    assert generate(tmp_path, compiled.model, prompt, len(trace) - len(prompt)) == trace


def test_match_heads_on_random_prompts(tmp_path):
    lookup = match(
        [read_at(0), read_at(1)], [history(1), history(0)], period=3, phase=2, start=5, sentinel=5, digit_limit=3
    )
    model = compile_machine(Machine(read_at(lookup - 2), 0, 3, min_prefix=6), max_context=64)
    to_huggingface(model, tmp_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(tmp_path)
    rng = random.Random(0)
    for _ in range(40):
        prompt = [rng.randint(0, 3) for _ in range(3 + 3 * rng.randint(1, 6) + rng.randint(0, 2))]
        logits = hf(torch.tensor([prompt_ids(model, prompt)])).logits[0, -1]
        assert tokens_from_ids(model, [int(logits.argmax())]) == model.generate(prompt, 1)[-1:]


def test_life_through_a_text_generation_pipeline(tmp_path):
    spec = LifeSpec(width=6, height=5, generations=4)
    model = compile_machine(spec.machine())
    to_huggingface(model, tmp_path)
    prompt = spec.encode(place(spec, GLIDER, 1, 1))
    generate_text = transformers.pipeline("text-generation", model=str(tmp_path))
    text = generate_text(
        " ".join(map(str, prompt)),
        max_new_tokens=spec.max_context - spec.cells,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]
    generated = [int(t) for t in text.split()]
    assert generated == model.generate(prompt, spec.max_context - spec.cells)[len(prompt) :]


def test_values_too_large_for_float32_are_refused(tmp_path):
    model = compile_machine(Machine((history() + 1).clip(0, 4000), 0, 4000), max_context=8000)
    with pytest.raises(ValueError, match="2\\*\\*24"):
        to_huggingface(model, tmp_path)
