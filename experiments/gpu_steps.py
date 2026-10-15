"""Time serial generation per token on a GPU and on the CPU decoder, by model size.

usage: python experiments/gpu_steps.py [--sizes 4,16,64,256,1024,2048]

Synthetic @program functions with k assignments in a loop body: width grows
with k, depth barely does (it grows with log k). Prints one row per
model: layers, nonzero weights, and microseconds per token on each device.
"""

import argparse
import importlib.util
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from firebrand.gpu_generate import GPUGenerator


def synthetic(k, directory, variables=8, seed=0):
    rng = random.Random(seed)
    lines = [
        "from firebrand.program import program, Int",
        "",
        "@program(max_context=1 << 20)",
        "def synthetic(seed: Int[0, 255]):",
    ]
    lines += [f"    v{j}: Int[0, 255] = (seed + {j * 37}) % 256" for j in range(variables)]
    lines.append("    for step in range(1000):")
    for _ in range(k):
        a, b, c = (rng.randrange(variables) for _ in range(3))
        lines.append(f"        v{a} = (v{b} + v{c} + {rng.randrange(256)}) % 256")
    lines.append("    return v0")
    path = Path(directory) / f"synthetic_{k}.py"
    path.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location(f"synthetic_{k}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic


def weights(model):
    return sum(int(np.count_nonzero(w.vals if hasattr(w, "vals") else w)) for w in model.weights.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="4,16,64,256,1024,2048")
    args = parser.parse_args()
    print(f"{'statements':>10} {'layers':>7} {'weights':>9} {'GPU us':>8} {'CPU us':>8}")
    with tempfile.TemporaryDirectory() as directory:
        for k in map(int, args.sizes.split(",")):
            program = synthetic(k, directory)
            model, prompt = program.model, program.prompt(7)
            gpu = GPUGenerator(model).step_time(prompt, 3000)
            model.generate(prompt, 50, backend="c")
            start = time.time()
            out = model.generate(prompt, 3000, backend="c")
            cpu = (time.time() - start) / (len(out) - len(prompt))
            print(f"{k:>10} {model.config['layers']:>7} {weights(model):>9,} {gpu * 1e6:>8.0f} {cpu * 1e6:>8.0f}")


if __name__ == "__main__":
    main()
