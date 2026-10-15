"""Run DOOM in the transformer for a few contexts, checking every record against the emulator.

usage: run_doom_contexts.py [--contexts N] [--frames F] [--context 2**K] [--json FILE] [--save DIR]

Each context ends with the model writing the next context's RAM, registers, and
pc; the host copies them into the next prompt. Records, console text, and frame
checksums must equal the reference emulator's.
Build: make -C riscv/doom ZONE_MB=3 RAM_TOP=4718592 FAST_DRAW=1
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from firebrand import compile_machine  # noqa: E402
from firebrand.machines.rv32 import RV32Spec  # noqa: E402

RAM = 4718592
ELF = ROOT / f"riscv/doom/build_O2_mb3_ram{RAM}_fastdraw/doom.elf"
WAD = ROOT / "third_party/doom1.wad"
EMULATOR = ROOT / "riscv/emulator/rv32emu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, default=2)
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames (0: no limit)")
    parser.add_argument("--save", help="directory for frames as PNG")
    parser.add_argument("--context", type=int, default=1 << 25)
    parser.add_argument("--json")
    args = parser.parse_args()

    rom = WAD.read_bytes()
    rom += bytes(-len(rom) % 4)
    spec = RV32Spec(ram_bytes=RAM, rom_bytes=len(rom), max_context=args.context, continuation=True)
    image, entry = spec.load_elf(ELF)
    capacity = (spec.dump_start - spec.protocol.prefix_length) // spec.record.width
    print(f"prompt {spec.protocol.prefix_length:,} tokens; up to {capacity:,} instructions per context; "
          f"dump {spec.dump_bytes:,} tokens", flush=True)
    model = compile_machine(spec.machine())

    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory, "trace")
        frame_log = Path(directory, "frames.csv")
        subprocess.run([str(EMULATOR), "--rom", str(WAD), "--max-instructions", str(args.contexts * capacity),
                        "--trace", str(reference), "--frame-log", str(frame_log), str(ELF)],
                       capture_output=True, check=False)
        expected = open(reference)
        reference_frames = [line.split(",")[-1].strip() for line in frame_log.read_text().splitlines()[1:]]
        prompt = spec.encode(image, entry, rom)
        contexts, console, executed, ok, frames = [], "", 0, True, []
        for number in range(args.contexts):
            start = time.time()
            trace = model.generate(prompt, spec.max_context - len(prompt), backend="c")
            seconds = time.time() - start
            records = spec.records(trace)
            wanted = [tuple(map(int, next(expected).split())) for _ in records]
            matches = records == wanted
            ok &= matches
            for event in spec.events(trace, prompt[:RAM], rom):
                if event[0] == "write":
                    console += event[1].decode("latin-1")
                else:
                    frames.append(event[1:])
                    if args.save:
                        save_png(Path(args.save) / f"frame_{len(frames) - 1:05d}.png", *event[1:])
            executed += len(records)
            contexts.append({"context": number + 1, "instructions": len(records), "seconds": round(seconds),
                             "records_match": matches, "continued": spec.continues(trace),
                             "frames_so_far": len(frames)})
            print(json.dumps(contexts[-1]), flush=True)
            if not matches or not spec.continues(trace) or (args.frames and len(frames) >= args.frames):
                break
            prompt = spec.next_prompt(trace, rom)

    emulator = subprocess.run([str(EMULATOR), "--rom", str(WAD), "--max-instructions", str(executed), str(ELF)],
                              capture_output=True, check=False)
    checksums = [f"{fnv1a(pixels):016x}" for pixels, _ in frames]
    result = {
        "instructions": executed,
        "contexts": contexts,
        "records_match": ok,
        "console_matches": console == emulator.stdout.decode("latin-1"),
        "frames": len(frames),
        "frames_match": checksums == reference_frames[:len(checksums)],
        "frame_checksums": checksums,
        "console": console,
    }
    print(console, end="")
    print(json.dumps({k: v for k, v in result.items() if k != "console"}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    sys.exit(0 if ok and result["console_matches"] and result["frames_match"] else 1)


def fnv1a(data):
    """The emulator's frame checksum (FNV-1a, 64-bit) over the palette indices."""
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def save_png(path, pixels, palette, scale=2):
    """A frame as PNG: each palette index mapped through the (b, g, r, a) palette."""
    sys.path.insert(0, str(ROOT / "riscv/tools"))
    from ppm2png import write_png
    rgb = bytes(c for p in pixels for c in (palette[4 * p + 2], palette[4 * p + 1], palette[4 * p]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(str(path), 320, 200, rgb, scale)


if __name__ == "__main__":
    main()
