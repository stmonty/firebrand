"""Run RV32IM programs through the compiled transformer and check them in lockstep.

usage: run_transformer_tests.py [--jobs N] [--json FILE] [--backend c|python] ELF...

For each ELF (linked with riscv/runtime/tiny.ld), the reference emulator writes
its per-instruction records (next pc, write key, value). The transformer runs
from the same RAM image; every generated record must equal the emulator's, and
the program must halt at its ECALL. Prints one line per program plus a summary.
"""

import os

# One BLAS thread per worker process; the pool supplies the parallelism.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EMULATOR = ROOT / "riscv/emulator/rv32emu"
sys.path.insert(0, str(ROOT / "src"))

from firebrand import Transformer, compile_machine  # noqa: E402
from firebrand.machines.rv32 import RV32Spec  # noqa: E402

_model = None
CONTEXT = 1 << 17
BACKEND = "c"


def reference(elf):
    """Emulator records, process status, and how many divisions executed."""
    with tempfile.NamedTemporaryFile("r", suffix=".trace") as trace, \
            tempfile.NamedTemporaryFile("r", suffix=".json") as summary:
        run = subprocess.run([str(EMULATOR), "--trace", trace.name, "--summary", summary.name, str(elf)],
                             capture_output=True)
        records = [tuple(map(int, line.split())) for line in trace]
        ops = json.load(summary)["ops"]
    return records, run.returncode, sum(ops[o] for o in ("div", "divu", "rem", "remu"))


def check(args):
    elf, model_path, backend = args
    global _model, BACKEND
    BACKEND = backend
    spec = RV32Spec(max_context=CONTEXT)
    if _model is None:
        _model = Transformer.load(model_path)
    expected, status, divisions = reference(elf)
    image, entry = spec.load_elf(elf)
    prefix = spec.encode(image, entry)
    start = time.time()
    # Each division takes 64 extra micro-step records.
    trace = _model.generate(prefix, spec.protocol.token_budget(len(expected) + 1 + 64 * divisions),
                            backend=BACKEND)
    seconds = time.time() - start
    records = spec.records(trace)
    # The emulator also records the final ECALL; the transformer halts instead.
    mismatch = next((i for i, (a, b) in enumerate(zip(records, expected)) if a != b), None)
    registers = spec.registers(trace)
    return {
        "program": Path(elf).stem,
        "instructions": len(expected),
        "divisions": divisions,
        "tokens": len(trace) - len(prefix),
        "seconds": round(seconds, 1),
        "halted": spec.halted(trace),
        "records_match": mismatch is None and len(records) == len(expected) - 1,
        "first_mismatch": mismatch,
        "exit_code": registers[10],
        "emulator_status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elfs", nargs="+")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--json")
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    args = parser.parse_args()
    global BACKEND
    BACKEND = args.backend
    spec = RV32Spec(max_context=CONTEXT)
    with tempfile.TemporaryDirectory() as directory:
        compile_machine(spec.machine()).save(directory)
        with ProcessPoolExecutor(args.jobs) as pool:
            results = []
            for result in pool.map(check, [(e, directory, BACKEND) for e in args.elfs]):
                ok = result["halted"] and result["records_match"] and result["exit_code"] == 0
                print(f"{'PASS' if ok else 'FAIL'} {result['program']}: {result['instructions']} instructions, "
                      f"{result['tokens']} tokens, {result['seconds']}s"
                      + ("" if ok else f" {result}"), flush=True)
                results.append(dict(result, passed=ok))
    passed = sum(r["passed"] for r in results)
    print(f"{passed}/{len(results)} passed")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
