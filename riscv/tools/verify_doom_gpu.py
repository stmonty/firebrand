"""Run DOOM's whole demo through the transformer on a GPU, by speculative decoding.

usage: verify_doom_gpu.py [--contexts N] [--chunk C] [--save-frames] [--out DIR]

The emulator drafts every token of each context (rv32emu --draft) and the GPU
checks them all against the transformer at once. Each next prompt is cut from
the checked trace, and console text and frames decoded from the checked traces
must equal the emulator's.
Build: make -C riscv/doom ZONE_MB=3 RAM_TOP=4718592 FAST_DRAW=1
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from firebrand import compile_machine  # noqa: E402
from firebrand.gpu import Verifier  # noqa: E402
from firebrand.machines.rv32 import CALL_FRAME, CALL_WRITE, MEMORY, ROM_BASE, RV32Spec  # noqa: E402

RAM = 4718592
CONTEXT = 1 << 25
ELF = ROOT / f"riscv/doom/build_O2_mb3_ram{RAM}_fastdraw/doom.elf"
WAD = ROOT / "third_party/doom1.wad"
EMULATOR = ROOT / "riscv/emulator/rv32emu"
OUT = ROOT / "artifacts/doom/gpu_run"


def fnv1a(data):
    """The emulator's frame checksum (FNV-1a, 64-bit) over the palette indices."""
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def events(spec, trace, rom):
    """Console writes and frames of one checked context, decoded from its trace by
    replaying register and memory writes (spec.events, vectorized): a list of
    ("write", bytes) and ("frame", pixels, palette) in order."""
    width, prefix = spec.record.width, spec.protocol.prefix_length
    continues = len(trace) == spec.end + 1 and trace[-1] == spec.protocol.halt_token
    end = spec.dump_start if continues else len(trace) - 1          # A stopped program ends with HALT.
    rows = trace[prefix:end].astype(np.int64).reshape(-1, width)
    next_pc = rows[:, 0] | rows[:, 1] << 8 | rows[:, 2] << 16
    value = rows[:, 3] | rows[:, 4] << 8 | rows[:, 5] << 16 | rows[:, 6] << 24
    key = rows[:, 7] | rows[:, 8] << 8 | rows[:, 9] << 16
    resume = trace[prefix - width:prefix - width + 3].astype(np.int64)
    executing = np.concatenate([[resume[0] | resume[1] << 8 | resume[2] << 16], next_pc[:-1]])
    ram = trace[:spec.ram_bytes].astype(np.uint8)
    code = ram[:len(ram) // 4 * 4].view("<u4")
    calls = np.flatnonzero(code[np.minimum(executing, len(code) - 1)] == 0x73)
    initial = [0] + [int(trace[spec.registers_start + r * width + 3:spec.registers_start + r * width + 7]
                         .astype(np.int64).dot([1, 256, 65536, 16777216])) for r in range(1, 32)]

    def register(r, at):
        writes = np.flatnonzero(key == r)
        found = np.searchsorted(writes, at) - 1
        return np.where(found >= 0, value[writes[np.maximum(found, 0)]], initial[r])

    a0, a1, a2, a7 = (register(r, calls) for r in (10, 11, 12, 17))
    memory = ram.copy().view("<u4")
    stores = np.flatnonzero(key >= MEMORY)
    applied, out = 0, []

    def span(address, length):
        if address >= ROM_BASE:
            return bytes(rom[address - ROM_BASE:address - ROM_BASE + length])
        return memory.view(np.uint8)[address:address + length].tobytes()

    for call, x0, x1, x2, x7 in zip(calls, a0, a1, a2, a7):
        wanted = (x7 == CALL_WRITE and x0 in (1, 2)) or x7 == CALL_FRAME
        if not wanted:
            continue
        upto = np.searchsorted(stores, call)
        if upto > applied:            # Apply writes before this call; the newest write to a word wins.
            words, values = key[stores[applied:upto]] - MEMORY, value[stores[applied:upto]]
            last = len(words) - 1 - np.unique(words[::-1], return_index=True)[1]
            memory[words[last]] = values[last].astype(np.uint32)
            applied = upto
        if x7 == CALL_WRITE:
            out.append(("write", span(int(x1), int(x2))))
        else:
            out.append(("frame", span(int(x0), 320 * 200), span(int(x1), 1024)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contexts", type=int, help="stop after N contexts")
    parser.add_argument("--chunk", type=int, default=1 << 19, help="positions per GPU chunk")
    parser.add_argument("--save-frames", action="store_true", help="write PNG frames under the output directory")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output directory (default {OUT.relative_to(ROOT)})")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rom = WAD.read_bytes()
    rom += bytes(-len(rom) % 4)
    rom_tokens = np.frombuffer(rom, dtype=np.uint8).astype(np.int16)
    spec = RV32Spec(ram_bytes=RAM, rom_bytes=len(rom), max_context=CONTEXT, continuation=True)
    started = time.time()
    model = compile_machine(spec.machine())
    verifier = Verifier(model, chunk=args.chunk)
    capacity = (spec.dump_start - spec.protocol.prefix_length) // spec.record.width
    image, entry = spec.load_elf(ELF)

    with tempfile.TemporaryDirectory() as directory:
        fifo = Path(directory) / "draft"
        os.mkfifo(fifo)
        console_path, frames_path = out / "emulator_console.txt", out / "emulator_frames.csv"
        command = [str(EMULATOR), "--rom", str(WAD), "--context-records", str(capacity), "--state-bytes", str(RAM),
                   "--draft", str(fifo), "--frame-log", str(frames_path), "--merged-output", "--trace-exit"]
        emulator = subprocess.Popen(command + [str(ELF)], stdout=open(console_path, "wb"), stderr=subprocess.PIPE)
        stream = open(fifo, "rb", buffering=1 << 24)

        # Drafts are read and turned into context traces ahead of the GPU. Each
        # context's prompt is cut from the previous context's trace: the host
        # copies the dump the transformer generates (once the GPU has checked it).
        drafts = queue.Queue(maxsize=2)
        pad = np.zeros(spec.registers_start - spec.ram_bytes - spec.rom_bytes, dtype=np.int16)

        def reader():
            prompt = spec.encode_array(image, entry, rom)
            for records, state in spec.read_drafts(stream, RAM):
                trace = spec.draft_trace(prompt, records, state, rom)
                drafts.put((prompt, records, state, trace))
                if state is not None:
                    prompt = np.concatenate([trace[spec.dump_start:spec.dump_start + spec.ram_bytes], rom_tokens,
                                             pad, trace[spec.registers_out:spec.end]])
            drafts.put(None)

        threading.Thread(target=reader, daemon=True).start()
        decoded, decoders = [], []
        log, executed, number, ok = open(out / "contexts.jsonl", "w"), 0, 0, True
        while (item := drafts.get()) is not None:
            prompt, records, state, trace = item
            number += 1
            begun = time.time()
            wrong, token = verifier.check(torch.from_numpy(trace), len(prompt))
            seconds = time.time() - begun
            if wrong is not None:
                ok = False
                print(json.dumps({"context": number, "mismatch_at": wrong, "drafted": int(trace[wrong]),
                                  "transformer": token}), flush=True)
                break
            # Decode console text and frames from the checked trace meanwhile.
            worker = threading.Thread(target=lambda t=trace, n=number: decoded.append((n, events(spec, t, rom))))
            worker.start()
            decoders.append(worker)
            # Division micro-steps (scratch keys) are not instructions.
            executed += int(np.sum((records[:, 7] < 32) | (records[:, 7] >= MEMORY)
                                   | (records[:, 8] > 0) | (records[:, 9] > 0)))
            entry_log = {"context": number, "tokens": len(trace) - len(prompt), "gpu_seconds": round(seconds, 2),
                         "instructions": executed, "hours": round((time.time() - started) / 3600, 3)}
            log.write(json.dumps(entry_log) + "\n")
            log.flush()
            if number % 10 == 0 or state is None:
                print(json.dumps(entry_log), flush=True)
            if state is None or (args.contexts and number >= args.contexts):
                break
        for worker in decoders:
            worker.join()
        if args.contexts and state is not None:
            emulator.kill()
        emulator.wait()
        stderr = emulator.stderr.read().decode()

    console, checksums = "", []
    for _, items in sorted(decoded):
        for event in items:
            if event[0] == "write":
                console += event[1].decode("latin-1")
            else:
                checksums.append(f"{fnv1a(event[1]):016x}")
                if args.save_frames:
                    sys.path.insert(0, str(ROOT / "riscv/tools"))
                    from run_doom_contexts import save_png
                    save_png(out / "frames" / f"frame_{len(checksums) - 1:05d}.png", *event[1:])
    finished = state is None and ok
    if not finished:
        # A partial run: the reference is the emulator stopped after the same instructions.
        with open(console_path, "wb") as console_file:
            subprocess.run([str(EMULATOR), "--rom", str(WAD), "--max-instructions", str(executed), "--frame-log",
                            str(frames_path), "--merged-output", str(ELF)], stdout=console_file, check=False)
    reference = [line.split(",")[-1].strip() for line in frames_path.read_text().splitlines()[1:]]
    summary = {
        "contexts": number, "instructions": executed, "all_tokens_checked": ok, "halted": finished,
        "frames": len(checksums),
        "frames_match": checksums == reference,
        "console_matches": console == console_path.read_bytes().decode("latin-1"),
        "hours": round((time.time() - started) / 3600, 3), "chunk": args.chunk,
        "gpu": torch.cuda.get_device_name(0), "emulator_exit": stderr.strip().splitlines()[-1:] if stderr else [],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if ok and summary["frames_match"] and summary["console_matches"] else 1)


if __name__ == "__main__":
    main()
