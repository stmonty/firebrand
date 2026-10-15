"""Read machine states saved by rv32emu --save-states (format: rv32emu.c, save_state)."""

import struct

MAGIC = 0x31767273


def read_state(path):
    """{instructions, contexts, clock_us, frames, keys, pc, registers, ram} of a saved state:
    the state just before the first instruction of context `contexts` + 1."""
    data = open(path, "rb").read()
    magic, ram_bytes = struct.unpack_from("<2I", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an rv32emu state")
    instructions, contexts, clock_us, frames = struct.unpack_from("<4Q", data, 8)
    keys, pc = struct.unpack_from("<iI", data, 40)
    registers = list(struct.unpack_from("<32I", data, 48))
    ram = data[176:176 + ram_bytes]
    if len(ram) != ram_bytes:
        raise ValueError(f"{path}: truncated")
    return dict(instructions=instructions, contexts=contexts, clock_us=clock_us, frames=frames,
                keys=keys, pc=pc, registers=registers, ram=ram)
