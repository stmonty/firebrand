"""The RV32IM interpreter compiled into weights.

A toolchain-free test assembles a small program directly. When the RISC-V build
outputs exist (see docs/doom.md), conformance tests and a C program also
run in lockstep with the reference emulator.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from firebrand import Transformer, compile_machine, fast
from firebrand.machines.rv32 import RV32Spec

ROOT = Path(__file__).resolve().parents[1]
EMULATOR = ROOT / "riscv/emulator/rv32emu"
TESTS = ROOT / "riscv/tests/build_tiny"
PROGRAMS = ROOT / "riscv/programs/build"


@pytest.fixture(scope="module")
def rv32():
    spec = RV32Spec(max_context=1 << 17)
    return spec, compile_machine(spec.machine())


# -- A tiny assembler for the handful of instructions the test program uses ------


def r_type(funct7, rs2, rs1, funct3, rd, opcode=0x33):
    return funct7 << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode


def i_type(imm, rs1, funct3, rd, opcode=0x13):
    return (imm & 0xFFF) << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode


def s_type(imm, rs2, rs1, funct3=2):
    return (imm >> 5 & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | (imm & 31) << 7 | 0x23


def b_type(imm, rs2, rs1, funct3):
    return (
        (imm >> 12 & 1) << 31
        | (imm >> 5 & 0x3F) << 25
        | rs2 << 20
        | rs1 << 15
        | funct3 << 12
        | (imm >> 1 & 0xF) << 8
        | (imm >> 11 & 1) << 7
        | 0x63
    )


ADD, SUB, MUL, DIV, DIVU, REM = ((0, 0), (0x20, 0), (1, 0), (1, 4), (1, 5), (1, 6))
MASK32 = 0xFFFFFFFF
ECALL = 0x73

# Sum 1..10 in a loop, store and reload it, then multiply and divide.
PROGRAM = [
    i_type(10, 0, 0, 5),  # 0:  t0 = 10
    i_type(0, 0, 0, 6),  # 4:  t1 = 0
    r_type(ADD[0], 5, 6, ADD[1], 6),  # 8:  t1 += t0
    i_type(-1, 5, 0, 5),  # 12: t0 -= 1
    b_type(-8, 0, 5, 1),  # 16: bne t0, zero, 8
    s_type(1024, 6, 0),  # 20: mem[1024] = t1
    i_type(1024, 0, 2, 7, 0x03),  # 24: t2 = mem[1024]
    r_type(MUL[0], 7, 7, MUL[1], 28),  # 28: t3 = t2 * t2 = 3025
    i_type(-7, 0, 0, 29),  # 32: t4 = -7
    r_type(DIV[0], 29, 28, DIV[1], 30),  # 36: t5 = 3025 / -7 = -432
    r_type(REM[0], 29, 28, REM[1], 31),  # 40: t6 = 3025 % -7 = 1
    r_type(DIVU[0], 29, 28, DIVU[1], 10),  # 44: a0 = 3025 / 0xFFFFFFF9 = 0
    ECALL,  # 48
]


def test_assembled_program(rv32):
    spec, model = rv32
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(PROGRAM):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    prefix = spec.encode(image, 0)
    budget = spec.protocol.token_budget(60 + 3 * 65)
    trace = model.generate(prefix, budget)
    assert model.generate(prefix, budget, backend="c") == trace
    # The compiled decoder read memoized byte lookups back instead of rerunning them.
    assert fast.statistics()[2] > 0
    assert spec.halted(trace)
    registers = spec.registers(trace)
    assert registers[6] == registers[7] == 55
    assert registers[28] == 3025
    assert registers[30] == -432 & 0xFFFFFFFF
    assert registers[31] == 1
    assert registers[10] == 0
    records = spec.records(trace)
    assert records[32][1:] == (256 + 96, 55)  # The store names memory word 256.
    assert len(spec.records(trace, micro=True)) == len(records) + 3 * 64


def lockstep(spec, model, elf):
    """Every transformer record must equal the emulator's record for that instruction."""
    trace_path, summary_path = (Path(str(elf) + suffix) for suffix in (".trace", ".json"))
    subprocess.run(
        [str(EMULATOR), "--trace", str(trace_path), "--summary", str(summary_path), str(elf)],
        check=False,
        capture_output=True,
    )
    expected = [tuple(map(int, line.split())) for line in trace_path.read_text().splitlines()]
    ops = json.loads(summary_path.read_text())["ops"]
    divisions = sum(ops[o] for o in ("div", "divu", "rem", "remu"))
    trace_path.unlink()
    summary_path.unlink()
    image, entry = spec.load_elf(elf)
    trace = model.generate(
        spec.encode(image, entry), spec.protocol.token_budget(len(expected) + 1 + 64 * divisions), backend="c"
    )
    assert spec.halted(trace)
    assert spec.records(trace) == expected[:-1]  # The emulator also logs the ECALL.
    return spec.registers(trace)


needs_build = pytest.mark.skipif(
    not EMULATOR.exists() or not TESTS.exists(), reason="RISC-V toolchain outputs not built"
)


@needs_build
@pytest.mark.parametrize(
    "elf", sorted(TESTS.glob("*.elf")) + sorted(PROGRAMS.glob("*.elf")), ids=lambda path: path.stem
)
def test_programs_in_lockstep(rv32, elf):
    """All 48 RV32IM conformance tests and the C programs, record by record."""
    spec, model = rv32
    registers = lockstep(spec, model, elf)
    # gcd60 and primes60 exit with their result; the others exit(0) after checking themselves.
    expected = {"gcd60": 12, "primes60": 17}.get(elf.stem, 0)
    assert registers[10] == expected and registers[17] == 93


def test_checkpoint_reload(rv32, tmp_path):
    spec, model = rv32
    model.save(tmp_path)
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(PROGRAM[:5] + [ECALL]):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    prefix = spec.encode(image, 0)
    budget = spec.protocol.token_budget(40)
    assert Transformer.load(tmp_path).generate(prefix, budget) == model.generate(prefix, budget)


def test_host_calls_and_console(rv32):
    """write returns its length and continues; frame continues; exit halts."""
    spec, model = rv32
    text = b"hi!\n"
    program = [
        i_type(64, 0, 0, 17),  # a7 = 64 (write)
        i_type(1, 0, 0, 10),  # a0 = 1 (stdout)
        i_type(512, 0, 0, 11),  # a1 = 512 (the text)
        i_type(len(text), 0, 0, 12),  # a2 = 4
        ECALL,  # write: a0 = 4
        0x10 << 12 | 17 << 7 | 0x37,  # lui a7, 0x10 (frame call 0x10000)
        ECALL,  # frame: no register changes
        i_type(93, 0, 0, 17),  # a7 = 93 (exit)
        ECALL,
    ]
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(program):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    image[512 : 512 + len(text)] = text
    trace = model.generate(spec.encode(image, 0), spec.protocol.token_budget(20), backend="c")
    assert spec.halted(trace)
    assert len(spec.records(trace)) == 8  # Everything but the final ECALL.
    assert spec.registers(trace)[10] == len(text)
    assert spec.console(trace, image) == "hi!\n"
    # The frame call's framebuffer pointer is a0, which write set to 4.
    frames = [e for e in spec.events(trace, image) if e[0] == "frame"]
    assert len(frames) == 1 and frames[0][1][:4] == bytes(image[4:8]) and len(frames[0][1]) == 64000


def run_contexts(spec, model, prompt, backend="c"):
    """Run context after context; the host only copies each dump into the next prompt."""
    records, contexts = [], 0
    while True:
        contexts += 1
        trace = model.generate(prompt, spec.max_context - len(prompt), backend=backend)
        records += spec.records(trace)
        if not spec.continues(trace):
            return records, trace, contexts
        prompt = spec.next_prompt(trace)


@pytest.fixture(scope="module")
def continued():
    spec = RV32Spec(max_context=1 << 15, continuation=True)
    return spec, compile_machine(spec.machine())


@needs_build
@pytest.mark.parametrize("name", ["fib", "fixed", "crc32", "rv32um-div", "rv32ui-sw"])
def test_programs_across_contexts(continued, name):
    spec, model = continued
    elf = (PROGRAMS if not name.startswith("rv32") else TESTS) / f"{name}.elf"
    trace_path = Path(str(elf) + ".continued")
    subprocess.run([str(EMULATOR), "--trace", str(trace_path), str(elf)], check=False, capture_output=True)
    expected = [tuple(map(int, line.split())) for line in trace_path.read_text().splitlines()]
    trace_path.unlink()
    image, entry = spec.load_elf(elf)
    records, trace, contexts = run_contexts(spec, model, spec.encode(image, entry))
    assert spec.halted(trace) and spec.registers(trace)[10] == 0
    assert records == expected[:-1]
    assert contexts >= (2 if len(expected) > 1500 else 1)


@needs_build
@pytest.mark.parametrize("name", ["fixed", "crc32", "fib"])
def test_emulator_states_equal_the_models_carried_state(continued, name, tmp_path):
    """The emulator predicts every context boundary, and a prompt encoded from its
    saved state equals the prompt the model wrote for the next context, so a long
    run can be split into segments that start from saved states."""
    sys.path.insert(0, str(ROOT / "riscv/tools"))
    from emulator_state import read_state

    spec, model = continued
    elf = (PROGRAMS if not name.startswith("rv32") else TESTS) / f"{name}.elf"
    capacity = (spec.dump_start - spec.protocol.prefix_length) // spec.record.width
    subprocess.run(
        [
            str(EMULATOR),
            "--context-records",
            str(capacity),
            "--save-states",
            str(tmp_path),
            "--state-bytes",
            str(spec.ram_bytes),
            str(elf),
        ],
        check=False,
        capture_output=True,
    )
    image, entry = spec.load_elf(elf)
    prompt, executed, boundaries = spec.encode(image, entry), 0, 0
    while True:
        trace = model.generate(prompt, spec.max_context - len(prompt), backend="c")
        executed += len(spec.records(trace))
        if not spec.continues(trace):
            break
        prompt = spec.next_prompt(trace)
        boundaries += 1
        state = read_state(tmp_path / f"state_{boundaries:06d}.bin")
        assert state["instructions"] == executed
        assert spec.encode(state["ram"], state["pc"], registers=state["registers"]) == prompt
    assert boundaries >= 1 and not (tmp_path / f"state_{boundaries + 1:06d}.bin").exists()


def test_division_cut_by_a_context_boundary_restarts(continued):
    """All divisions: each takes 65 records, so boundaries land inside them."""
    spec, model = continued
    program = [
        i_type(40, 0, 0, 5),  # t0 = 40 iterations
        i_type(1000, 0, 0, 6),  # t1 = 1000
        i_type(-7, 0, 0, 29),  # t4 = -7
        r_type(DIV[0], 29, 6, DIV[1], 30),  # loop: t5 = t1 / t4
        r_type(REM[0], 29, 6, REM[1], 31),  # t6 = t1 % t4
        r_type(0, 31, 6, 0, 6),  # t1 += t6 (changes the next dividend)
        i_type(-1, 5, 0, 5),  # t0 -= 1
        b_type(-16, 0, 5, 1),  # bne t0, zero, loop
        i_type(93, 0, 0, 17),  # exit
        ECALL,
    ]
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(program):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    _records, trace, contexts = run_contexts(spec, model, spec.encode(image, 0))
    t1 = 1000
    for _ in range(40):
        quotient = int(t1 / -7)  # Truncates toward zero.
        remainder = t1 - quotient * -7
        t1 += remainder
    registers = spec.registers(trace)
    assert contexts >= 3 and spec.halted(trace)
    assert (registers[30], registers[31], registers[6]) == (quotient & MASK32, remainder & MASK32, t1 & MASK32)


def test_numpy_and_c_agree_across_contexts():
    spec = RV32Spec(ram_bytes=512, max_context=2600, continuation=True)
    model = compile_machine(spec.machine())
    program = [
        i_type(30, 0, 0, 5),
        r_type(0, 5, 6, 0, 6),
        i_type(-1, 5, 0, 5),
        b_type(-8, 0, 5, 1),
        s_type(256, 6, 0),
        i_type(93, 0, 0, 17),
        ECALL,
    ]
    image = bytearray(spec.ram_bytes)
    for i, word in enumerate(program):
        image[4 * i : 4 * i + 4] = word.to_bytes(4, "little")
    prompt = spec.encode(image, 0)
    c_records, c_trace, contexts = run_contexts(spec, model, prompt)
    numpy_records, numpy_trace, _ = run_contexts(spec, model, prompt, backend="python")
    assert contexts >= 2 and c_records == numpy_records and c_trace == numpy_trace
    assert spec.registers(c_trace)[6] == sum(range(1, 31))


@pytest.fixture(scope="module")
def short_contexts():
    """Contexts of about 300 records: divisions (65 records) are often cut short."""
    spec = RV32Spec(max_context=20000, continuation=True)
    return spec, compile_machine(spec.machine())


@needs_build
@pytest.mark.parametrize("name", ["fixed", "crc32", "rv32um-div"])
@pytest.mark.parametrize("contexts_of", ["continued", "short_contexts"])
def test_emulator_drafts_equal_the_models_traces(contexts_of, name, tmp_path, request):
    """rv32emu --draft predicts every token of every context, including division
    micro-steps, divisions cut short by a boundary, and the RAM dump, so a GPU
    can check a whole context in parallel instead of generating it."""
    spec, model = request.getfixturevalue(contexts_of)
    elf = (PROGRAMS if not name.startswith("rv32") else TESTS) / f"{name}.elf"
    capacity = (spec.dump_start - spec.protocol.prefix_length) // spec.record.width
    subprocess.run(
        [
            str(EMULATOR),
            "--context-records",
            str(capacity),
            "--state-bytes",
            str(spec.ram_bytes),
            "--draft",
            str(tmp_path / "draft"),
            str(elf),
        ],
        check=False,
        capture_output=True,
    )
    image, entry = spec.load_elf(elf)
    prompt = spec.encode(image, entry)
    assert spec.encode_array(image, entry).tolist() == prompt
    contexts = 0
    with open(tmp_path / "draft", "rb") as stream:
        for records, state in spec.read_drafts(stream, spec.ram_bytes):
            contexts += 1
            trace = model.generate(prompt, spec.max_context - len(prompt), backend="c")
            assert spec.draft_trace(prompt, records, state).tolist() == trace
            if state is None:
                break
            prompt = spec.next_prompt(trace)
            assert spec.encode_array(state[2], state[0], registers=state[1]).tolist() == prompt
    assert state is None and contexts >= (2 if name != "rv32um-div" or contexts_of == "short_contexts" else 1)
