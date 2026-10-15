"""32-bit word circuits against Python integer semantics, then as compiled weights."""

import random

import pytest

from firebrand import Machine, compile_machine, position_mod, read_at, select
from firebrand import words as w
from firebrand.reference import evaluate_all

MASK = 0xFFFFFFFF
EDGES = [
    0,
    1,
    2,
    3,
    7,
    31,
    32,
    255,
    256,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x10000,
    0x12345678,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
    0xDEADBEEF,
]


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def truncating_div(a, b):
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def rv_div(a, b):
    if b == 0:
        return MASK
    if signed(a) == -(2**31) and signed(b) == -1:
        return a
    return truncating_div(signed(a), signed(b)) & MASK


def rv_rem(a, b):
    if b == 0:
        return a
    if signed(a) == -(2**31) and signed(b) == -1:
        return 0
    return (signed(a) - truncating_div(signed(a), signed(b)) * signed(b)) & MASK


A = w.from_bytes(read_at(i) for i in range(4))
B = w.from_bytes(read_at(4 + i) for i in range(4))
SHIFT = w.shift_amount(B)

OPERATIONS = {
    "add": (lambda: w.add(A, B)[0], lambda a, b: (a + b) & MASK),
    "sub": (lambda: w.sub(A, B)[0], lambda a, b: (a - b) & MASK),
    "sltu": (lambda: w.ltu(A, B), lambda a, b: int(a < b)),
    "slt": (lambda: w.lt(A, B), lambda a, b: int(signed(a) < signed(b))),
    "eq": (lambda: w.equal(A, B), lambda a, b: int(a == b)),
    "and": (lambda: w.bitwise_and(A, B), lambda a, b: a & b),
    "or": (lambda: w.bitwise_or(A, B), lambda a, b: a | b),
    "xor": (lambda: w.bitwise_xor(A, B), lambda a, b: a ^ b),
    "sll": (lambda: w.shift_left(A, SHIFT), lambda a, b: (a << (b & 31)) & MASK),
    "srl": (lambda: w.shift_right(A, SHIFT), lambda a, b: a >> (b & 31)),
    "sra": (lambda: w.shift_right(A, SHIFT, arithmetic=True), lambda a, b: (signed(a) >> (b & 31)) & MASK),
    "mul": (lambda: w.multiply(A, B), lambda a, b: (a * b) & MASK),
    "mulhu": (lambda: w.multiply(A, B, high_word=True), lambda a, b: (a * b) >> 32),
    "mulh": (lambda: w.mulh(A, B), lambda a, b: ((signed(a) * signed(b)) >> 32) & MASK),
    "mulhsu": (lambda: w.mulh(A, B, signed_b=False), lambda a, b: ((signed(a) * b) >> 32) & MASK),
    "divu": (lambda: w.divide_unsigned(A, B)[0], lambda a, b: a // b if b else MASK),
    "remu": (lambda: w.divide_unsigned(A, B)[1], lambda a, b: a % b if b else a),
    "div": (lambda: w.divide_signed(A, B)[0], rv_div),
    "rem": (lambda: w.divide_signed(A, B)[1], rv_rem),
}
BUILT = {}


def built(name):
    if name not in BUILT:
        BUILT[name] = OPERATIONS[name][0]()
    return BUILT[name]


def encode(a, b):
    return [a >> (8 * i) & 255 for i in range(4)] + [b >> (8 * i) & 255 for i in range(4)]


def value(result, tokens):
    if isinstance(result, w.Word):
        out = evaluate_all(result.bytes, tokens)
        return sum(byte << (8 * i) for i, byte in enumerate(out))
    return evaluate_all([result], tokens)[0]


def pairs(count, seed):
    rng = random.Random(seed)
    edge = [(a, b) for a in EDGES for b in EDGES]
    return (
        edge
        + [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(count)]
        + [(rng.getrandbits(32), rng.getrandbits(rng.randint(1, 16))) for _ in range(count)]
    )


@pytest.mark.parametrize("name", OPERATIONS)
def test_circuit_matches_python(name):
    result, reference = built(name), OPERATIONS[name][1]
    count = 30 if name in ("div", "rem", "divu", "remu") else 150
    for a, b in pairs(count, hash(name) & 0xFFFF):
        assert value(result, encode(a, b)) == reference(a, b), (name, hex(a), hex(b))


@pytest.mark.parametrize("name", ["add", "slt", "xor", "sra", "mul", "mulh", "divu"])
def test_compiled_circuit_emits_result_bytes(name):
    result = built(name)
    out = result.bytes if isinstance(result, w.Word) else (result, 0, 0, 0)
    phase = position_mod(4, 1)
    emitted = select(phase.equals(0), out[0], select(phase.equals(1), out[1], select(phase.equals(2), out[2], out[3])))
    model = compile_machine(Machine(emitted, 0, 255, min_prefix=8), max_context=64)
    reference = OPERATIONS[name][1]
    for a, b in pairs(4, 7)[::53][:12]:
        prefix = encode(a, b)
        expected = reference(a, b)
        # Each generated token reads the inputs at positions 0..7, never the output.
        assert model.generate(prefix, 4)[8:] == [expected >> (8 * i) & 255 for i in range(4)]
