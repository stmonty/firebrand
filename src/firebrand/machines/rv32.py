"""An RV32IM interpreter compiled into transformer weights.

Tokens are bytes (0..255) plus HALT (-1). The prompt holds the RAM image (RAM
address a is position a), an optional read-only ROM mapped at 0x40000000, the
registers, and a resume record. Each instruction emits one record:

    [pc0, pc1, pc2, v0, v1, v2, v3, k0, k1, k2]

the next pc (word index) and a key naming what was written: 0 nothing, 1..31 a
register, 32..95 division scratch, 96 + w memory word w; v0..v3 are the value.
Registers and memory are read with `match` heads (the newest record with that
key), falling back to the prompt's image, ROM, or zero.

ECALL: exit (93) halts, write (64) returns its length in a0, frame (0x10000)
continues; the host reads text and frames from memory replayed from the trace.
Division is microcoded as 64 records of restoring division (partial remainder
and quotient under scratch keys) and a final record.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import words as w
from ..ir import Machine, constant, history, integer, match, position_digit, position_quotient, read_at, select
from ..machine import RecordMachine, Transition
from ..records import Field, Record

LOAD, MISC_MEM, OP_IMM, AUIPC, STORE, OP, LUI, BRANCH, JALR, JAL, SYSTEM = (
    0x03,
    0x0F,
    0x13,
    0x17,
    0x23,
    0x33,
    0x37,
    0x63,
    0x67,
    0x6F,
    0x73,
)
ROM_BASE = 0x40000000
SCRATCH, MEMORY = 32, 96  # First scratch key; first memory-word key.
CALL_EXIT, CALL_WRITE, CALL_FRAME = 93, 64, 0x10000
PC_DIGITS = ("p0", "p1", "p2")
VALUE = ("v0", "v1", "v2", "v3")
KEY_DIGITS = ("k0", "k1", "k2")
_SHIFT2 = tuple(v >> 2 for v in range(256))
_LOW2 = tuple(v & 3 for v in range(256))
_SIGN = tuple(v >> 7 for v in range(256))
_LOW6_TIMES4 = tuple((v & 63) * 4 for v in range(256))
_SHIFT6 = tuple(v >> 6 for v in range(256))


@dataclass(frozen=True)
class RV32Spec:
    ram_bytes: int = 8192  # Addressable RAM; memory keys cover all of it.
    image_bytes: int | None = None  # Initialized RAM in the prompt (default: all of RAM).
    rom_bytes: int = 0  # Read-only memory at ROM_BASE, after the image.
    max_context: int = 1 << 17
    continuation: bool = False  # Dump RAM and registers for the next context.

    def __post_init__(self):
        if self.image_bytes is None:
            object.__setattr__(self, "image_bytes", self.ram_bytes)
        for value in (self.ram_bytes, self.image_bytes, self.rom_bytes):
            integer(value)
        if self.ram_bytes % 4 or not 256 <= self.ram_bytes <= 1 << 25:
            raise ValueError("ram_bytes must be a multiple of 4 up to 32 MiB")  # Keys fit 24 bits.
        if self.image_bytes % 4 or not 4 <= self.image_bytes <= self.ram_bytes:
            raise ValueError("image_bytes must be a positive multiple of 4 within RAM")
        if self.rom_bytes % 4 or not 0 <= self.rom_bytes <= 1 << 24:
            raise ValueError("rom_bytes must be a multiple of 4 up to 16 MiB")
        _ = self.protocol  # Validates the alphabet and context.
        if self.continuation:
            if self.image_bytes != self.ram_bytes:
                raise ValueError("continuation carries all of RAM: image_bytes must equal ram_bytes")
            if self.dump_start < self.protocol.prefix_length + self.record.width:
                raise ValueError("context too small to execute anything before its dump")

    @property
    def record(self):
        return Record("RV32", tuple(Field(name, 0, 255) for name in PC_DIGITS + VALUE + KEY_DIGITS))

    # Prompt layout: [RAM image][ROM][pad][sentinel record][31 register records]
    # [resume record]. The sentinel record comes first, so every lookup (including
    # those in prefill passes over the register records) has a fallback. With
    # continuation, a context's generated tokens end with the same pieces for the
    # next context: [executed records][RAM dump][sentinel record][31 register
    # records][resume record][HALT], at fixed positions.

    @property
    def registers_start(self):
        width = self.record.width
        return -(-(self.image_bytes + self.rom_bytes) // width) * width

    @property
    def protocol(self):
        prefix = self.registers_start + 33 * self.record.width
        return RecordMachine(self.record, prefix, self.max_context, -1, 255)

    @property
    def sentinel(self):
        """The sentinel record's key position: lookups with no match return it."""
        return self.registers_start + self.record.width - 1

    @property
    def dump_bytes(self):
        return -(-self.ram_bytes // self.record.width) * self.record.width

    @property
    def dump_start(self):
        """First position of the RAM dump: the last record boundary that leaves
        room for the dump, the register records, the resume record, and HALT."""
        width, prefix = self.record.width, self.protocol.prefix_length
        room = self.max_context - 1 - 33 * width - self.dump_bytes - prefix
        return prefix + room // width * width

    @property
    def registers_out(self):
        return self.dump_start + self.dump_bytes

    @property
    def end(self):
        """Position of the HALT that ends a continued context."""
        return self.registers_out + 33 * self.record.width

    # -- Host side: serialization and reading results (no execution) --------------

    def encode(self, image, entry, rom=b"", registers=None):
        """The first prompt: image, ROM, register records (zero unless given), and
        a resume record at the entry point."""
        image, rom = bytes(image), bytes(rom)
        if len(image) > self.image_bytes or len(rom) > self.rom_bytes:
            raise ValueError("image or ROM larger than configured")
        if entry % 4 or not 0 <= entry < self.image_bytes:
            raise ValueError("entry point must be an aligned address in the image")
        registers = list(registers or [0] * 32)
        payload = list(image) + [0] * (self.image_bytes - len(image)) + list(rom) + [0] * (self.rom_bytes - len(rom))
        payload += [0] * (self.registers_start + self.record.width - len(payload))  # Sentinel.
        for r in range(1, 32):
            payload += [0, 0, 0, *(registers[r] >> 8 * k & 255 for k in range(4)), r, 0, 0]
        pc = entry // 4
        return self.protocol.initialize(
            payload, p0=pc & 255, p1=pc >> 8 & 255, p2=pc >> 16, **{name: 0 for name in VALUE + KEY_DIGITS}
        )

    def continues(self, trace):
        """Whether a context ended by writing the next context's state."""
        return self.continuation and len(trace) == self.end + 1 and trace[-1] == self.protocol.halt_token

    def next_prompt(self, trace, rom=b""):
        """The next context's prompt, copied from this context's generated dump:
        the dumped RAM becomes the image, then the ROM, then the register and
        resume records. The host only moves tokens."""
        if not self.continues(trace):
            raise ValueError("trace did not end with a continuation")
        rom = list(bytes(rom)) + [0] * (self.rom_bytes - len(rom))
        image = trace[self.dump_start : self.dump_start + self.ram_bytes]
        pad = [0] * (self.registers_start - self.ram_bytes - self.rom_bytes)
        return image + rom + pad + trace[self.registers_out : self.end]

    def encode_array(self, image, pc, rom=b"", registers=None):
        """encode() as an int16 NumPy array (fast for large prompts)."""
        image, rom = bytes(image), bytes(rom)
        if len(image) > self.image_bytes or len(rom) > self.rom_bytes:
            raise ValueError("image or ROM larger than configured")
        if pc % 4 or not 0 <= pc < self.image_bytes:
            raise ValueError("entry point must be an aligned address in the image")
        width = self.record.width
        out = np.zeros(self.protocol.prefix_length, dtype=np.int16)
        out[: len(image)] = np.frombuffer(image, dtype=np.uint8)
        out[self.image_bytes : self.image_bytes + len(rom)] = np.frombuffer(rom, dtype=np.uint8)
        registers = list(registers or [0] * 32)
        for r in range(1, 32):
            base = self.registers_start + r * width
            out[base + 3 : base + 7] = [registers[r] >> 8 * k & 255 for k in range(4)]
            out[base + 7] = r
        word = pc // 4
        out[-width:] = self.record.pack(
            p0=word & 255, p1=word >> 8 & 255, p2=word >> 16, **{name: 0 for name in VALUE + KEY_DIGITS}
        )
        return out

    @staticmethod
    def read_drafts(stream, ram_bytes):
        """Contexts from an `rv32emu --draft` stream (see rv32emu.c): (records, state)
        per context, with records an [n, 10] uint8 array and state (pc, registers,
        RAM) where the context continues, or None where the program stopped."""

        def read(n):
            data = stream.read(n)
            if len(data) != n:
                raise ValueError("truncated draft")
            return data

        while True:
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError("truncated draft")
            count = int.from_bytes(header, "little")
            records = np.frombuffer(read(10 * count), dtype=np.uint8).reshape(count, 10)
            state = None
            if read(1)[0]:
                pc = int.from_bytes(read(4), "little")
                registers = np.frombuffer(read(128), dtype="<u4").tolist()
                state = (pc, registers, read(ram_bytes))
            yield records, state

    def draft_trace(self, prompt, records, state, rom=b""):
        """The whole trace a context would hold if the transformer generates what the
        emulator drafted: the prompt, the records, and then either HALT (the program
        stopped) or the RAM dump, the register and resume records for the next
        context, and HALT. An int16 array; the transformer checks it (gpu.py)."""
        halt = np.array([self.protocol.halt_token], dtype=np.int16)
        parts = [np.asarray(prompt, dtype=np.int16), records.reshape(-1).astype(np.int16)]
        if state is None:
            return np.concatenate(parts + [halt])
        if len(prompt) + records.size != self.dump_start:
            raise ValueError("a continuing draft must fill the context up to the dump")
        pc, registers, ram = state
        dump = np.zeros(self.dump_bytes, dtype=np.int16)
        dump[: self.ram_bytes] = np.frombuffer(ram, dtype=np.uint8)
        tail = self.encode_array(b"", pc, registers=registers)[self.registers_start :]
        return np.concatenate(parts + [dump, tail, halt])

    def load_elf(self, path):
        """Initialized RAM image (up to the last loaded byte) and entry point."""
        data = Path(path).read_bytes()
        if data[:4] != b"\x7fELF" or data[4] != 1 or int.from_bytes(data[18:20], "little") != 243:
            raise ValueError("not an RV32 ELF file")
        entry = int.from_bytes(data[24:28], "little")
        phoff, phentsize, phnum = (int.from_bytes(data[a:b], "little") for a, b in ((28, 32), (42, 44), (44, 46)))
        image = bytearray()
        for i in range(phnum):
            header = data[phoff + i * phentsize : phoff + (i + 1) * phentsize]
            kind, offset, vaddr, _, filesz, memsz = (
                int.from_bytes(header[4 * k : 4 * k + 4], "little") for k in range(6)
            )
            if kind != 1:
                continue
            if vaddr + memsz > self.ram_bytes:
                raise ValueError("segment does not fit the configured RAM")
            if vaddr + filesz > len(image):
                image.extend(bytes(vaddr + filesz - len(image)))
            image[vaddr : vaddr + filesz] = data[offset : offset + filesz]
        image.extend(bytes(-len(image) % 4))
        return bytes(image), entry

    def records(self, trace, micro=False):
        """(next_pc_word, key, value) per complete record; division micro-steps only if micro=True."""
        start, width = self.protocol.prefix_length, self.record.width
        end = min(len(trace), self.dump_start) if self.continuation else len(trace)
        body = np.asarray(trace[start:end], dtype=np.int64)
        halt = np.flatnonzero(body == self.protocol.halt_token)
        if halt.size:
            if halt[0] % width:
                raise ValueError("halt token inside a record")
            body = body[: halt[0]]
        rows = body[: len(body) // width * width].reshape(-1, width)
        pc = rows[:, 0] | rows[:, 1] << 8 | rows[:, 2] << 16
        value = rows[:, 3] | rows[:, 4] << 8 | rows[:, 5] << 16 | rows[:, 6] << 24
        key = rows[:, 7] | rows[:, 8] << 8 | rows[:, 9] << 16
        keep = np.ones(len(rows), bool) if micro else (key < SCRATCH) | (key >= MEMORY)
        return list(zip(pc[keep].tolist(), key[keep].tolist(), value[keep].tolist()))

    def _prompt_registers(self, trace):
        registers, width = [0] * 32, self.record.width
        for r in range(1, 32):
            base = self.registers_start + r * width
            registers[r] = sum(trace[base + 3 + k] << 8 * k for k in range(4))
        return registers

    def registers(self, trace):
        """Register file: the prompt's register records, then every write."""
        registers = self._prompt_registers(trace)
        for _, key, value in self.records(trace):
            if 0 < key < 32:
                registers[key] = value
        return registers

    def halted(self, trace):
        """Whether the program itself stopped (not a continuation)."""
        return trace[-1] == self.protocol.halt_token and not self.continues(trace)

    def events(self, trace, image, rom=b""):
        """Host-call output, read from memory replayed from the trace (the host
        adapter's whole job): ("write", bytes) for console text and ("frame",
        pixels, palette) for each frame call, with 320x200 palette indices and a
        256-entry (blue, green, red, alpha) palette. The executing pc of each
        record is the previous record's next pc."""
        image, rom = bytes(image), bytes(rom)
        start = self.protocol.prefix_length - self.record.width
        initial = trace[start : start + 3]
        pc = initial[0] | initial[1] << 8 | initial[2] << 16
        registers, memory = self._prompt_registers(trace), {}

        def byte(address):
            word = memory.get(address >> 2)
            if word is not None:
                return word >> 8 * (address & 3) & 255
            if address < len(image):
                return image[address]
            if 0 <= address - ROM_BASE < len(rom):
                return rom[address - ROM_BASE]
            return 0

        def span(address, length):
            return bytes(byte(address + i) for i in range(length))

        for next_pc, key, value in self.records(trace, micro=True):
            if int.from_bytes(image[4 * pc : 4 * pc + 4], "little") == 0x73:
                if registers[17] == CALL_WRITE and registers[10] in (1, 2):
                    yield "write", span(registers[11], registers[12])
                elif registers[17] == CALL_FRAME:
                    yield "frame", span(registers[10], 320 * 200), span(registers[11], 1024)
            if 0 < key < SCRATCH:
                registers[key] = value
            elif key >= MEMORY:
                memory[key - MEMORY] = value
            pc = next_pc

    def console(self, trace, image, rom=b""):
        """Text the program wrote to stdout/stderr (see events)."""
        return b"".join(e[1] for e in self.events(trace, image, rom) if e[0] == "write").decode("latin-1")

    # -- The compiled transition --------------------------------------------------

    def machine(self):
        shared = {}
        lowered = self.protocol.lower_machine(lambda step: self.transition(step, shared))
        if not self.continuation:
            return lowered
        # During the RAM dump every token is a RAM byte, not a record field.
        return Machine(
            select(shared["dumping"], shared["byte"], lowered.next_value),
            lowered.lower,
            lowered.upper,
            lowered.halt_values,
            min_prefix=lowered.min_prefix,
            context_limit=lowered.context_limit,
        )

    def _lookup(self, digits):
        """Newest record whose key tokens equal digits: (position of its last key
        token, whether it matched rather than returning the sentinel). Keys run
        from the prompt's register records up to the dump (if any)."""
        width = self.record.width
        at = match(
            digits,
            [history(2), history(1), history(0)],
            period=width,
            phase=width - 1,
            start=self.registers_start,
            sentinel=self.sentinel,
            end=self.dump_start if self.continuation else None,
        )
        return at, 1 - (at - self.sentinel).equals(0)

    @staticmethod
    def _stored(at):
        """The value bytes of the record whose last token is at `at`."""
        return w.Word(tuple(read_at(at - 6 + k).clip(0, 255) for k in range(4)))

    def transition(self, step, shared=None):
        zero = constant(0)
        width = self.record.width
        if self.continuation:
            # Which part of the context this record is in, from its position alone.
            def reached(position):
                return (step.address - (position - 1)).clip(0, 1)

            executing = 1 - reached(self.dump_start)
            dumping = (reached(self.dump_start) - reached(self.registers_out)).clip(0, 1)
            # Record 0 after the dump is the sentinel record (register 0 reads as
            # zero, key zero), then registers 1..31, then the resume record.
            register_dump = (reached(self.registers_out) - reached(self.registers_out + 32 * width)).clip(0, 1)
            resuming = (reached(self.registers_out + 32 * width) - reached(self.end)).clip(0, 1)
            dumped_register = position_quotient(width, 1 - self.registers_out).clip(0, 31)
            dump_address = w.Word(tuple(position_digit(k, 1 - self.dump_start) for k in range(3)) + (0,))
        # Previous-record fields are bytes; the clip states that (the alphabet includes -1).
        digits = [step.state[name].clip(0, 255) for name in PC_DIGITS]
        pc = (digits[0] + 256 * digits[1] + 65536 * digits[2]).clip(0, self.image_bytes // 4 - 1)
        instruction = w.Word(tuple(read_at(4 * pc + k).clip(0, 255) for k in range(4)))
        ib = w.bits(instruction)

        def field(lo, hi):
            total = zero
            for i in range(lo, hi):
                total = total + ib[i] * (1 << (i - lo))
            return total

        opcode, rd, funct3 = field(0, 7), field(7, 12), field(12, 15)
        is_ = {
            code: opcode.equals(code)
            for code in (LOAD, MISC_MEM, OP_IMM, AUIPC, STORE, OP, LUI, BRANCH, JALR, JAL, SYSTEM)
        }
        # ECALL reads a7 (x17) and a2 (x12) through the same two register lookups.
        rs1 = select(is_[SYSTEM], 17, field(15, 20))
        if self.continuation:
            rs1 = select(register_dump, dumped_register, rs1)
        rs2 = select(is_[SYSTEM], 12, field(20, 25))
        f3 = [funct3.equals(k) for k in range(8)]
        sign = ib[31]
        imm_i = w.from_bits([ib[20 + i] if i < 12 else sign for i in range(32)])
        imm_s = w.from_bits([ib[7 + i] if i < 5 else ib[20 + i] if i < 12 else sign for i in range(32)])
        imm_b = w.from_bits(
            [
                zero if i == 0 else ib[7 + i] if i < 5 else ib[20 + i] if i < 11 else ib[7] if i == 11 else sign
                for i in range(32)
            ]
        )
        imm_u = w.from_bits([zero if i < 12 else ib[i] for i in range(32)])
        imm_j = w.from_bits(
            [
                zero if i == 0 else ib[20 + i] if i < 11 else ib[20] if i == 11 else ib[i] if i < 20 else sign
                for i in range(32)
            ]
        )

        def register(index):
            at, found = self._lookup([index, zero, zero])
            live = select(index.equals(0), 0, found)
            return w.Word(tuple(select(live, b, 0) for b in self._stored(at).bytes))

        a, b_register = register(rs1), register(rs2)
        # Register-register operations and branches compare against rs2; OP-IMM uses imm_i.
        b = w.select_word(_any(is_[OP], is_[BRANCH]), b_register, imm_i)

        # Operand isolation: a functional unit sees its operands only when the
        # decoded instruction uses it, and zeros otherwise. Results are selected
        # by opcode anyway, so outputs are unchanged; but an idle unit's inputs
        # stay constant, and the incremental decoder skips it.
        m_extension = select(is_[OP], field(25, 32).equals(1), 0)
        arithmetic = select(m_extension, 0, _any(is_[OP], is_[OP_IMM]))
        shifting = select(arithmetic, funct3.lookup((0, 1, 0, 0, 0, 1, 0, 0)), 0)
        logical = select(arithmetic, funct3.lookup((0, 0, 0, 0, 1, 0, 1, 1)), 0)
        multiplying = select(m_extension, funct3.lookup((1, 1, 1, 1, 0, 0, 0, 0)), 0)
        division = select(m_extension, funct3.lookup((0, 0, 0, 0, 1, 1, 1, 1)), 0)
        memory_access = _any(is_[LOAD], is_[STORE], *([dumping] if self.continuation else []))

        def gate(flag, word):
            return w.select_word(flag, word, w.word(0))

        # Memory: one lookup serves both loads (read) and stores (read-modify-write).
        address = w.add(a, w.select_word(is_[STORE], imm_s, imm_i))[0]
        if self.continuation:
            address = w.select_word(dumping, dump_address, address)
        address = gate(memory_access, address)
        memory_key = _key_digits(_word_index(address), MEMORY)
        at, found = self._lookup(memory_key)
        fallback = self._prompt_word(address)
        old = [select(found, s, f) for s, f in zip(self._stored(at).bytes, fallback)]
        lane = address.bytes[0].lookup(_LOW2)
        lanes = [lane.equals(k) for k in range(4)]
        byte = select(lanes[0], old[0], select(lanes[1], old[1], select(lanes[2], old[2], old[3])))
        half = (select(lanes[2], old[2], old[0]), select(lanes[2], old[3], old[1]))
        byte_fill, half_fill = 255 * byte.lookup(_SIGN), 255 * half[1].lookup(_SIGN)
        loaded = _by_funct3(
            f3,
            {
                0: w.Word((byte, byte_fill, byte_fill, byte_fill)),
                1: w.Word((*half, half_fill, half_fill)),
                2: w.Word(tuple(old)),
                4: w.Word((byte, 0, 0, 0)),
                5: w.Word((*half, 0, 0)),
            },
        )
        source = gate(is_[STORE], b_register).bytes
        stored = _by_funct3(
            f3,
            {
                0: w.Word(tuple(select(lanes[k], source[0], old[k]) for k in range(4))),
                1: w.Word(
                    (
                        select(lanes[2], old[0], source[0]),
                        select(lanes[2], old[1], source[1]),
                        select(lanes[2], source[0], old[2]),
                        select(lanes[2], source[1], old[3]),
                    )
                ),
                2: w.Word(source),
            },
        )

        # Integer ALU: every unit runs; funct3/funct7 select one result.
        alternate = ib[30]
        added = w.add(a, b)[0]
        subtracting = select(arithmetic, select(is_[OP], alternate, 0), 0)
        subtracted = w.sub(gate(subtracting, a), gate(subtracting, b))[0]
        # Comparisons serve SLT/SLTU (funct3 2 and 3) and branches.
        comparing = _any(select(arithmetic, funct3.lookup((0, 0, 1, 1, 0, 0, 0, 0)), 0), is_[BRANCH])
        compare_a, compare_b = gate(comparing, a), gate(comparing, b)
        less, less_unsigned = w.lt(compare_a, compare_b), w.ltu(compare_a, compare_b)
        equal = w.equal(compare_a, compare_b)
        shift_bits = w.bits(gate(shifting, a))
        amount = w.shift_amount(gate(shifting, b))
        logic_a, logic_b = w.bits(gate(logical, a)), w.bits(gate(logical, b))
        alu = _by_funct3(
            f3,
            {
                0: w.select_word(select(is_[OP], alternate, 0), subtracted, added),
                1: w.shift_left(shift_bits, amount),
                2: w.Word((less, 0, 0, 0)),
                3: w.Word((less_unsigned, 0, 0, 0)),
                4: w.bitwise_xor(logic_a, logic_b),
                5: w.select_word(
                    alternate, w.shift_right(shift_bits, amount, arithmetic=True), w.shift_right(shift_bits, amount)
                ),
                6: w.bitwise_or(logic_a, logic_b),
                7: w.bitwise_and(logic_a, logic_b),
            },
        )
        multiplicand, multiplier = gate(multiplying, a), gate(multiplying, b)
        low, high = w.multiply_both(multiplicand, multiplier)
        micro_step, micro_key, micro_value, divided = self._division(
            step, gate(division, a), gate(division, b_register), funct3, division
        )
        extension = {
            0: low,
            1: w.mulh(multiplicand, multiplier, high=high),
            2: w.mulh(multiplicand, multiplier, signed_b=False, high=high),
            3: high,
            4: divided,
            5: divided,
            6: divided,
            7: divided,
        }
        alu = w.select_word(m_extension, _by_funct3(f3, extension), alu)
        # A division stays at its pc, writing scratch keys, until its final record.
        continuing = select(division, micro_step, 0)

        # Host calls: a7 selects exit, write (a0 = length), frame, or unsupported.
        call = a.bytes
        high_zero = (call[1] + call[2] + call[3]).equals(0)
        call_write = select(is_[SYSTEM], select(high_zero, call[0].equals(CALL_WRITE), 0), 0)
        call_frame = select(is_[SYSTEM], ((call[0] + call[1] + call[3]).equals(0) + call[2].equals(1)).equals(2), 0)
        # ECALL is exactly 0x00000073; EBREAK and other SYSTEM encodings halt.
        ecall = select(
            is_[SYSTEM],
            (instruction.bytes[0] - 0x73 + instruction.bytes[1] + instruction.bytes[2] + instruction.bytes[3]).equals(
                0
            ),
            0,
        )
        continuing_call = select(ecall, _any(call_write, call_frame), 0)

        # Program counter as a byte address, and control transfer targets.
        pc_bytes = w.Word(
            (
                digits[0].lookup(_LOW6_TIMES4),
                digits[0].lookup(_SHIFT6) + digits[1].lookup(_LOW6_TIMES4),
                digits[1].lookup(_SHIFT6) + digits[2].lookup(_LOW6_TIMES4),
                digits[2].lookup(_SHIFT6),
            )
        )
        link = w.add(pc_bytes, w.word(4))[0]
        branch_base, jump_base, upper_base = (
            gate(is_[BRANCH], pc_bytes),
            gate(is_[JAL], pc_bytes),
            gate(is_[AUIPC], pc_bytes),
        )
        taken = _by_funct3(
            f3, {0: equal, 1: 1 - equal, 4: less, 5: 1 - less, 6: less_unsigned, 7: 1 - less_unsigned}, word=False
        )
        target = w.select_word(
            is_[JAL],
            w.add(jump_base, gate(is_[JAL], imm_j))[0],
            w.select_word(
                is_[JALR],
                added,
                w.select_word(
                    select(is_[BRANCH], taken, 0),
                    w.add(branch_base, gate(is_[BRANCH], imm_b))[0],
                    w.select_word(continuing, pc_bytes, link),
                ),
            ),
        )
        next_pc = _key_digits(_word_index(target), 0)

        result = w.select_word(
            is_[LUI],
            imm_u,
            w.select_word(
                is_[AUIPC],
                w.add(upper_base, gate(is_[AUIPC], imm_u))[0],
                w.select_word(
                    _any(is_[JAL], is_[JALR]),
                    link,
                    w.select_word(is_[LOAD], loaded, w.select_word(call_write, b_register, alu)),
                ),
            ),
        )
        writes_register = _any(is_[LUI], is_[AUIPC], is_[JAL], is_[JALR], is_[LOAD], is_[OP_IMM], is_[OP])
        destination = select(call_write, 10, rd)
        register_write = select(destination.equals(0), 0, _any(writes_register, call_write))
        writes = _any(register_write, is_[STORE], continuing)
        value = w.select_word(
            continuing, micro_value, w.select_word(is_[STORE], stored, w.select_word(register_write, result, w.word(0)))
        )
        key = [
            select(continuing, m, select(register_write, r, select(is_[STORE], k, 0)))
            for m, r, k in zip(micro_key, (destination, 0, 0), memory_key)
        ]
        known = _any(*is_.values())
        halt = _any(select(is_[SYSTEM], 1 - continuing_call, 0), 1 - known)
        pc_out = next_pc
        value_out = [select(writes, byte_, 0) for byte_ in value.bytes]
        key_out = key
        if self.continuation:
            # Register records carry x1..x31; the resume record carries the pc of
            # the last executed record (a division cut short simply restarts).
            # (Reads are eager: before the dump that position is in the future, so
            # read position 0 instead; the value is only used when resuming.)
            resume_pc = [read_at(select(executing, 0, self.dump_start - width + k)).clip(0, 255) for k in range(3)]
            pc_out = [select(executing, p, select(resuming, q, 0)) for p, q in zip(next_pc, resume_pc)]
            value_out = [select(executing, v, select(register_dump, r, 0)) for v, r in zip(value_out, a.bytes)]
            key_out = [
                select(executing, k, select(register_dump, r, 0)) for k, r in zip(key, (dumped_register, zero, zero))
            ]
            halt = select(executing, halt, reached(self.end))
            shared.update(dumping=dumping, byte=byte)
        return Transition(
            dict(
                **{name: d for name, d in zip(PC_DIGITS, pc_out)},
                **{name: v for name, v in zip(VALUE, value_out)},
                **{name: d for name, d in zip(KEY_DIGITS, key_out)},
            ),
            halt=halt,
        )

    def _prompt_word(self, address):
        """The aligned word containing address, from the RAM image or the ROM in
        the prompt; zero for RAM beyond the image. Every read position is valid
        even for addresses the program never uses (all reads are eager)."""
        b = address.bytes
        aligned0 = b[0].lookup(_SHIFT2) * 4
        offset = aligned0 + 256 * b[1] + 65536 * b[2]  # Low 24 bits, aligned.
        in_image = select(b[3].equals(0), w.ltu(address, w.word(self.image_bytes)), 0)
        last = self.image_bytes + self.rom_bytes - 4
        if self.rom_bytes:
            within = w.ltu(w.Word((b[0], b[1], b[2], 0)), w.word(self.rom_bytes))
            in_rom = select(b[3].equals(ROM_BASE >> 24), within, 0)
            base = select(in_rom, (offset + self.image_bytes).clip(0, last), offset.clip(0, last))
        else:
            in_rom = constant(0)
            base = offset.clip(0, last)
        present = _any(in_image, in_rom)
        return [select(present, read_at(base + k).clip(0, 255), 0) for k in range(4)]

    def _division(self, step, a, b, funct3, active):
        """One restoring-division micro-step, and the final signed result.

        Returns (continuing, key, value, result): whether this record is a
        micro-step (not the final record), its scratch key and value, and the
        RISC-V result the final record writes to rd.
        """
        width = self.record.width
        previous = step.state["k0"].clip(0, 255)
        higher = step.state["k1"].clip(0, 255) + step.state["k2"].clip(0, 255)
        # A micro-step follows a record whose key is a scratch key, 32..95.
        micro = select(
            higher.equals(0), ((previous - (SCRATCH - 1)).clip(0, 1) + (MEMORY - previous).clip(0, 1)).equals(2), 0
        )
        index = (previous - SCRATCH).clip(0, 63)  # 2i after A(i), 2i + 1 after Q(i).
        odd = index.lookup(tuple(v & 1 for v in range(64)))
        quotient_step = select(micro, 1 - odd, 0)  # A(i) was just written: write Q(i).
        final = select(micro, select(odd, index.equals(63), 0), 0)
        first = select(quotient_step, index.equals(0), 1 - micro)

        earlier = {}

        def back(records):
            # Expressions are shared by identity, so build each record's reads once.
            if records not in earlier:
                base = step.address - records * width
                # Isolated like the other units: zeros unless a division is running.
                earlier[records] = w.Word(
                    tuple(select(active, read_at(base + 3 + k).clip(0, 255), 0) for k in range(4))
                )
            return earlier[records]

        signed = funct3.lookup((0, 0, 0, 0, 1, 0, 1, 0))
        remainder_wanted = funct3.lookup((0, 0, 0, 0, 0, 0, 1, 1))
        negative_a = select(signed, w.sign(a), 0)
        negative_b = select(signed, w.sign(b), 0)
        dividend = w.select_word(negative_a, w.negate(a), a)
        divisor = w.select_word(negative_b, w.negate(b), b)

        partial = w.select_word(first, w.word(0), w.select_word(quotient_step, back(3), back(2)))
        quotient = w.select_word(first, dividend, w.select_word(quotient_step, back(2), back(1)))
        shifted = w._double_plus(list(partial.bytes), w.sign(quotient))  # Five bytes, < 2**33.
        difference, carry = [], constant(1)
        for x, y in zip(shifted, list(divisor.bytes) + [constant(0)]):
            low, carry = w.carry(x + (255 - y) + carry)
            difference.append(low)
        fits = carry  # No borrow: shifted >= divisor.
        remainder_out = w.Word(tuple(select(fits, d, x) for d, x in zip(difference[:4], shifted[:4])))
        quotient_out = w.Word(tuple(w._double_plus(list(quotient.bytes), fits)[:4]))
        value = w.select_word(quotient_step, quotient_out, remainder_out)
        key = [select(micro, (SCRATCH + index + 1).clip(SCRATCH, SCRATCH + 63), SCRATCH), constant(0), constant(0)]

        # Final record: Q(31) is one record back and A(31) two records back.
        q, r = back(1), back(2)
        zero_divisor = w.equal(b, w.word(0))
        flip = select(zero_divisor, 0, (negative_a + negative_b).equals(1))
        q = w.select_word(zero_divisor, w.select_word(signed, w.word(-1), q), w.select_word(flip, w.negate(q), q))
        r = w.select_word(negative_a, w.negate(r), r)
        return 1 - final, key, value, w.select_word(remainder_wanted, r, q)


def _word_index(address):
    """Three byte digits of (address >> 2), dropping bits above 24."""
    b = address.bytes
    return [b[i].lookup(_SHIFT2) + 64 * b[i + 1].lookup(_LOW2) for i in range(3)]


def _key_digits(digits, offset):
    """Digits of (number + offset) for a small constant offset, dropping overflow."""
    if not offset:
        return digits
    out, c = [], constant(offset)
    for d in digits:
        low, c = w.carry(d + c)
        out.append(low)
    return out


def _any(*conditions):
    """Logical OR of 0/1 conditions (the compiler needs the explicit 0..1 range)."""
    total = constant(0)
    for condition in conditions:
        total = total + condition
    return total.clip(0, 1)


def _by_funct3(f3, options, word=True):
    """Select among options keyed by funct3; unlisted values give zero."""
    result = w.word(0) if word else constant(0)
    for k, option in options.items():
        if word:
            result = w.select_word(f3[k], option, result)
        else:
            result = select(f3[k], option, result)
    return result
