"""A pretrained language model with a compiled exact-arithmetic stream.

usage: exact_arithmetic_lm.py                       # a few questions, with and without the stream
       exact_arithmetic_lm.py --evaluate [--per 20] [--only stream]
                                                    # accuracy by operand length -> artifacts/llm_arithmetic.json

The evaluation asks "What is A * B?" for random n-digit A and B and counts an
answer correct when the response contains the exact product. It compares the
0.5 B model with its compiled stream against the same model alone and against
larger models alone. Needs a CUDA GPU and `pip install transformers`.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

from firebrand.bridge import ExactArithmeticLM

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
BASELINES = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
SYSTEM = "When you compute something, first write the calculation as digits with an operator, like 12 * 34 = , then the result."
OUT = Path(__file__).resolve().parents[1] / "artifacts/llm_arithmetic.json"


def correct(text, a, b):
    return str(a * b) in re.sub(r"(?<=\d)[,\s](?=\d{3})", "", text)


def evaluate(per, only=None):
    rng = random.Random(0)
    lengths = [2, 3, 4, 6, 8, 10, 12]
    problems = {
        n: [(rng.randrange(10 ** (n - 1), 10**n), rng.randrange(10 ** (n - 1), 10**n)) for _ in range(per)]
        for n in lengths
    }
    results = json.loads(OUT.read_text())["results"] if only and OUT.exists() else {}

    def run(label, lm, use_stream):
        rows = {}
        for n in lengths:
            started, right, fired, samples = time.time(), 0, 0, []
            for a, b in problems[n]:
                new, forced = lm.generate(lm.chat_ids(f"What is {a} * {b}?", SYSTEM), 200, use_stream=use_stream)
                text = lm.tokenizer.decode(new, skip_special_tokens=True)
                right += correct(text, a, b)
                fired += forced > 0
                if len(samples) < 2:
                    samples.append({"a": a, "b": b, "response": text})
            rows[n] = {
                "correct": right,
                "of": per,
                "stream_used": fired,
                "seconds_per_answer": round((time.time() - started) / per, 2),
                "samples": samples,
            }
            print(
                f"{label:28} {n:2}-digit: {right}/{per} correct" + (f", stream used {fired}" if use_stream else ""),
                flush=True,
            )
        results[label] = rows

    lm = ExactArithmeticLM.from_pretrained(SMALL)
    run("Qwen2.5-0.5B + compiled stream", lm, True)
    if only == "stream":
        BASELINES.clear()
    else:
        run("Qwen2.5-0.5B", lm, False)
    del lm
    for name in BASELINES:
        lm = ExactArithmeticLM.from_pretrained(name)
        run(name.split("/")[1].replace("-Instruct", ""), lm, False)
        del lm
        torch.cuda.empty_cache()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"system_prompt": SYSTEM, "question": "What is A * B?", "per_length": per, "results": results}, indent=1
        )
        + "\n"
    )


def demo():
    lm = ExactArithmeticLM.from_pretrained(SMALL)
    for question in [
        "What is 48291 * 77153?",
        "A warehouse has 7354 boxes with 286 items each. How many items in total?",
        "Tell me a fun fact about octopuses.",
    ]:
        for use_stream in (False, True):
            label = "with the compiled stream" if use_stream else "language model alone"
            print(f"[{label}] {question}\n{lm.generate_text(question, SYSTEM, use_stream=use_stream)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--per", type=int, default=20, help="problems per operand length")
    parser.add_argument("--only", choices=["stream"], help="rerun only the model with the stream, keeping the rest")
    args = parser.parse_args()
    evaluate(args.per, args.only) if args.evaluate else demo()
