# DOOM

DOOM runs on an RV32IM interpreter compiled into transformer weights
(`firebrand.machines.rv32`). GCC compiles the engine for RV32IM; the transformer
executes the machine code one ten-token record per instruction, reading
registers and memory through attention.

## Pieces

| Piece | |
| --- | --- |
| Engine | [doomgeneric](https://github.com/ozkl/doomgeneric) at `dcb7a8d`, with a platform layer in [doomgeneric_rv32.c](../riscv/doom/doomgeneric_rv32.c) |
| Game data | Shareware `DOOM1.WAD` v1.9 (not included), run as `-timedemo demo1` |
| Toolchain | GCC with newlib built for exactly `rv32im`/`ilp32` ([toolchain.nix](../riscv/toolchain.nix)) |
| Reference | [rv32emu.c](../riscv/emulator/rv32emu.c); passes the user-level `rv32ui`/`rv32um` riscv-tests |
| Host calls | `exit`, `write` (console), `frame` (320×200 palette indices) |

The prompt holds DOOM's RAM image and the WAD as a read-only ROM. A 2^25-token
context holds about two million instructions. At a fixed position the model
writes its RAM, registers, and pc as tokens, and the host copies those into the
next context's prompt. The host also decodes console text and frames by
replaying the memory writes in the trace.

## Build

```sh
nix-build riscv/toolchain.nix -o third_party/rv32im-gcc
git clone https://github.com/ozkl/doomgeneric third_party/doomgeneric
git clone https://github.com/riscv-software-src/riscv-tests third_party/riscv-tests
cp /path/to/DOOM1.WAD third_party/doom1.wad
gcc -O2 -o riscv/emulator/rv32emu riscv/emulator/rv32emu.c
make -C riscv/programs
make -C riscv/tests
make -C riscv/doom ZONE_MB=3 RAM_TOP=4718592 FAST_DRAW=1
```

`ZONE_MB=3 RAM_TOP=4718592` shrinks DOOM's RAM to 4.5 MiB, which keeps each
context's RAM dump small; `FAST_DRAW=1` uses the drawing loops in
[draw_fast.c](../riscv/doom/draw_fast.c), which execute fewer instructions.

## Run

```sh
python riscv/tools/run_doom_contexts.py --contexts 12 --save frames/   # CPU, first frames
python riscv/tools/verify_doom_gpu.py --save-frames                    # whole demo on a GPU
python riscv/tools/run_doom_segments.py plan                           # whole demo on CPUs:
python riscv/tools/run_doom_segments.py run --workers 12               #   parallel segments
python riscv/tools/run_doom_segments.py summary
```

The GPU run lets the emulator draft each context (`rv32emu --draft`) and checks
every drafted token against the transformer in one parallel pass
(`firebrand.gpu`). The CPU runs generate every token with the compiled decoder.
Both check every record, console byte, and frame against the emulator.

## Results

The whole demo is 5,450,718,069 instructions in 2,948 contexts and 5,064 frames.
All frames and the console matched the emulator, through DOOM's exit.

| Run | Time |
| --- | --- |
| Compiled decoder, 12 CPU workers (Ryzen 7 5800X) | 9.2 h (111 CPU-hours) |
| Drafted and checked on one RTX 3080 Ti | 1.5 h |
