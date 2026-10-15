"""Run DOOM's whole demo in the transformer as parallel segments of contexts (CPU).

usage: run_doom_segments.py plan    [--every K]
       run_doom_segments.py run     [--workers W] [--segments A:B]
       run_doom_segments.py summary

plan saves the emulator's machine state at every K-th context boundary; run
starts each segment of K contexts from a saved state and checks it against the
emulator, including that its final carried prompt equals the next segment's
first; summary checks every segment and all frames.
Build: make -C riscv/doom ZONE_MB=3 RAM_TOP=4718592 FAST_DRAW=1
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "riscv/tools"))

from emulator_state import read_state  # noqa: E402
from run_doom_contexts import fnv1a, save_png  # noqa: E402
from firebrand import compile_machine  # noqa: E402
from firebrand.machines.rv32 import RV32Spec  # noqa: E402

RAM = 4718592
ELF = ROOT / f"riscv/doom/build_O2_mb3_ram{RAM}_fastdraw/doom.elf"
WAD = ROOT / "third_party/doom1.wad"
EMULATOR = ROOT / "riscv/emulator/rv32emu"
OUT = ROOT / "artifacts/doom/segments"
CONTEXT = 1 << 25


def rom():
    data = WAD.read_bytes()
    return data + bytes(-len(data) % 4)


def make_spec():
    return RV32Spec(ram_bytes=RAM, rom_bytes=len(rom()), max_context=CONTEXT, continuation=True)


def capacity(spec):
    return (spec.dump_start - spec.protocol.prefix_length) // spec.record.width


def plan(args):
    spec = make_spec()
    (OUT / "states").mkdir(parents=True, exist_ok=True)
    start = time.time()
    run = subprocess.run([str(EMULATOR), "--rom", str(WAD), "--context-records", str(capacity(spec)),
                          "--boundary-log", str(OUT / "boundaries.txt"), "--frame-log", str(OUT / "frames.csv"),
                          "--save-states", str(OUT / "states"), "--save-every", str(args.every),
                          "--state-bytes", str(RAM), "--trace-exit", "--merged-output", str(ELF)],
                         capture_output=True, check=False)
    (OUT / "console.txt").write_bytes(run.stdout)
    boundaries = [tuple(map(int, line.split())) for line in (OUT / "boundaries.txt").read_text().splitlines()]
    total = int(run.stderr.decode().split(" after ")[-1].split()[0])
    meta = {"every": args.every, "capacity": capacity(spec), "instructions": total,
            "contexts": len(boundaries) + 1, "segments": -(-(len(boundaries) + 1) // args.every),
            "exit": run.stderr.decode().strip().splitlines()[-1], "elf": str(ELF.relative_to(ROOT))}
    (OUT / "plan.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), f"\n({time.time() - start:.0f} s)")


def segment_bounds(meta, boundaries, s):
    """(first context, last context, instructions before it, instructions after it)."""
    first, last = s * meta["every"] + 1, min((s + 1) * meta["every"], meta["contexts"])
    before = 0 if s == 0 else boundaries[s * meta["every"] - 1][1]
    after = boundaries[last - 1][1] if last <= len(boundaries) else meta["instructions"]
    return first, last, before, after


def run_segment(s):
    os.environ["OMP_NUM_THREADS"] = "1"
    meta = json.loads((OUT / "plan.json").read_text())
    boundaries = [tuple(map(int, line.split())) for line in (OUT / "boundaries.txt").read_text().splitlines()]
    result_path = OUT / "results" / f"segment_{s:04d}.json"
    first, last, before, after = segment_bounds(meta, boundaries, s)
    spec, data = make_spec(), rom()
    model = compile_machine(spec.machine())
    started = time.time()
    if s == 0:
        image, entry = spec.load_elf(ELF)
        prompt, frame_base = spec.encode(image, entry, data), 0
    else:
        state = read_state(OUT / "states" / f"state_{first - 1:06d}.bin")
        assert state["instructions"] == before
        prompt = spec.encode(state["ram"], state["pc"], data, state["registers"])
        frame_base = state["frames"]

    with tempfile.TemporaryDirectory(dir=OUT) as directory:
        directory = Path(directory)
        command = [str(EMULATOR), "--rom", str(WAD), "--merged-output", "--max-instructions", str(after),
                   "--trace", str(directory / "trace"), "--frame-log", str(directory / "frames.csv")]
        if s:
            command += ["--load-state", str(OUT / "states" / f"state_{first - 1:06d}.bin")]
        reference = subprocess.run(command + [str(ELF)], capture_output=True, check=False)
        reference_frames = [line.split(",")[-1].strip()
                            for line in (directory / "frames.csv").read_text().splitlines()[1:]]
        expected = open(directory / "trace", "rb")
        contexts, console, frames, ok, executed, halted = [], "", [], True, before, False
        for number in range(first, last + 1):
            start = time.time()
            trace = model.generate(prompt, spec.max_context - len(prompt), backend="c")
            seconds = time.time() - start
            records = np.array(spec.records(trace), dtype=np.int64).reshape(-1, 3)
            lines = [expected.readline() for _ in range(len(records))]
            wanted = np.array(b" ".join(lines).split(), dtype=np.int64).reshape(-1, 3)
            matches = wanted.shape == records.shape and bool((wanted == records).all())
            ok &= matches
            for event in spec.events(trace, prompt[:RAM], data):
                if event[0] == "write":
                    console += event[1].decode("latin-1")
                else:
                    index = frame_base + len(frames)
                    frames.append(f"{fnv1a(event[1]):016x}")
                    save_png(OUT / "frames" / f"frame_{index:05d}.png", *event[1:])
            executed += len(records)
            halted = spec.halted(trace)
            contexts.append({"context": number, "instructions": len(records), "seconds": round(seconds),
                             "records_match": matches})
            if not matches or not spec.continues(trace):
                break
            prompt = spec.next_prompt(trace, data)
            del trace, records, wanted, lines
        expected.close()

    # The carried state must be the next segment's starting state.
    chained = None
    if ok and not halted:
        state = read_state(OUT / "states" / f"state_{last:06d}.bin")
        chained = (state["instructions"] == executed
                   and spec.encode(state["ram"], state["pc"], data, state["registers"]) == prompt)
    result = {
        "segment": s, "first_context": first, "last_context": last,
        "instructions_before": before, "instructions_after": executed, "expected_after": after,
        "records_match": ok, "halted": halted, "chains_to_next": chained,
        "console_matches": console == reference.stdout.decode("latin-1"),
        "frame_base": frame_base, "frame_checksums": frames,
        "frames_match": frames == reference_frames,
        "seconds": round(time.time() - started), "contexts": contexts, "console": console,
    }
    # A halt emits no record for the final exit call, which the emulator counts.
    result["ok"] = (ok and executed == after - halted and result["console_matches"] and result["frames_match"]
                    and (halted or chained))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    return s, result["ok"], result["seconds"], len(frames)


def run(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    meta = json.loads((OUT / "plan.json").read_text())
    low, high = (0, meta["segments"]) if not args.segments else map(int, args.segments.split(":"))
    done = {int(p.stem.split("_")[1]) for p in (OUT / "results").glob("segment_*.json")
            if json.loads(p.read_text())["ok"]} if (OUT / "results").exists() else set()
    todo = [s for s in range(low, high) if s not in done]
    print(f"{len(todo)} of {meta['segments']} segments to run on {args.workers} workers", flush=True)
    start = time.time()
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for count, (s, ok, seconds, frames) in enumerate(pool.imap_unordered(run_segment, todo), 1):
            elapsed = time.time() - start
            print(f"segment {s:4d}: {'ok' if ok else 'FAILED'} in {seconds} s, {frames} frames; "
                  f"{count}/{len(todo)} done, {elapsed / 3600:.2f} h elapsed, "
                  f"about {elapsed / count * (len(todo) - count) / 3600:.2f} h left", flush=True)


def summary(_args):
    meta = json.loads((OUT / "plan.json").read_text())
    results = [json.loads((OUT / "results" / f"segment_{s:04d}.json").read_text())
               if (OUT / "results" / f"segment_{s:04d}.json").exists() else None
               for s in range(meta["segments"])]
    missing = [s for s, r in enumerate(results) if r is None]
    failed = [r["segment"] for r in results if r and not r["ok"]]
    chain = all(r["instructions_before"] == (results[s - 1]["instructions_after"] if s else 0)
                for s, r in enumerate(results) if r and (s == 0 or results[s - 1]))
    checksums = [c for r in results if r for c in r["frame_checksums"]]
    reference = [line.split(",")[-1].strip() for line in (OUT / "frames.csv").read_text().splitlines()[1:]]
    console = "".join(r["console"] for r in results if r)
    out = {
        "segments": meta["segments"], "missing": missing, "failed": failed, "chained": chain,
        "contexts": sum(len(r["contexts"]) for r in results if r),
        "instructions": max((r["instructions_after"] for r in results if r), default=0),
        "frames": len(checksums), "reference_frames": len(reference),
        "frames_match": not missing and checksums == reference,
        "console_matches": not missing and console == (OUT / "console.txt").read_bytes().decode("latin-1"),
        "halted": bool(results[-1] and results[-1]["halted"]),
        "segment_cpu_hours": round(sum(r["seconds"] for r in results if r) / 3600, 2),
    }
    out["complete"] = (not missing and not failed and chain and out["frames_match"]
                       and out["console_matches"] and out["halted"])
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--every", type=int, default=10, help="contexts per segment")
    p = commands.add_parser("run")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--segments", help="A:B runs segments A..B-1 only")
    commands.add_parser("summary")
    args = parser.parse_args()
    {"plan": plan, "run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
