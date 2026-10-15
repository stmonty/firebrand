"""32-bit words as four byte expressions, with exact RV32IM arithmetic circuits.

A Word holds four little-endian byte expressions (0..255). Everything here builds
ordinary expressions, keeping intermediate values small: carries ripple between
bytes, bits come from nibble lookups, shifts are barrel shifters, products split
16-bit partial products into bytes, and division is restoring long division.
"""

from dataclasses import dataclass

from .ir import Expr, constant, integer, select

MASK = 0xFFFFFFFF
# Lookup tables are built once; large ones are shared by every node using them.
_HIGH_NIBBLE = tuple(v >> 4 for v in range(256))
_NIBBLE_BITS = tuple(tuple(v >> k & 1 for v in range(16)) for k in range(4))
_SIGN = tuple(v >> 7 for v in range(256))
_BIT4 = tuple(v >> 4 & 1 for v in range(256))
_PRODUCT_HIGH = tuple(v >> 8 for v in range(255 * 255 + 1))
_COLUMN_CARRY = tuple(v >> 8 for v in range(8 * 256))


@dataclass(frozen=True)
class Word:
    bytes: tuple  # Four expressions, least significant first, each 0..255.

    def __post_init__(self):
        if len(self.bytes) != 4:
            raise ValueError("a word has four bytes")
        object.__setattr__(self, "bytes", tuple(constant(b) for b in self.bytes))


def word(value):
    """A constant word; negative values wrap to two's complement."""
    value = integer(value) & MASK
    return Word(tuple(constant(value >> (8 * i) & 255) for i in range(4)))


def from_bytes(expressions):
    return Word(tuple(expressions))


# -- Bytes and bits ------------------------------------------------------------


def carry(total, base=256):
    """(total mod base, total div base) for an integer 0 <= total < 2 * base."""
    high = (total - (base - 1)).clip(0, 1)
    return _low(total, high, base), high


def _low(total, high, base=256):
    # Interval analysis cannot see that the low part is 0..base-1; the clip states
    # the exact range (and never changes the value).
    return (total - base * high).clip(0, base - 1)


def nibbles(byte):
    high = byte.lookup(_HIGH_NIBBLE)
    return _low(byte, high, 16), high


def bits(value):
    """Least-significant-first bits of a word (or a single byte expression)."""
    result = []
    for byte in value.bytes if isinstance(value, Word) else (value,):
        for nibble in nibbles(byte):
            result += [nibble.lookup(_NIBBLE_BITS[k]) for k in range(4)]
    return result


def from_bits(values):
    """Recombine least-significant-first bits into a Word (32) or bytes."""
    if len(values) % 8:
        raise ValueError("bit count must be a multiple of eight")
    bytes_ = []
    for i in range(0, len(values), 8):
        total = constant(0)
        for k, bit in enumerate(values[i : i + 8]):
            total = total + (bit * (1 << k) if k else bit)
        bytes_.append(total)
    return Word(tuple(bytes_)) if len(bytes_) == 4 else bytes_


# -- Addition and comparison ---------------------------------------------------


def add(a, b, carry_in=0):
    """(a + b + carry_in) mod 2**32 and the carry out."""
    out, c = [], constant(carry_in)
    for x, y in zip(a.bytes, b.bytes):
        low, c = carry(x + y + c)
        out.append(low)
    return Word(tuple(out)), c


def invert(a):
    return Word(tuple(255 - b for b in a.bytes))


def sub(a, b):
    """(a - b) mod 2**32 and the borrow (1 when a < b, unsigned)."""
    difference, c = add(a, invert(b), 1)
    return difference, 1 - c


def negate(a):
    return sub(word(0), a)[0]


def ltu(a, b):
    return sub(a, b)[1]


def sign(a):
    return a.bytes[3].lookup(_SIGN)


def flip_sign(a):
    """Map signed order onto unsigned order by toggling the top bit."""
    top = a.bytes[3]
    return Word(a.bytes[:3] + ((top + 128 - 256 * sign(a)).clip(0, 255),))


def lt(a, b):
    return ltu(flip_sign(a), flip_sign(b))


def equal(a, b):
    same = [(x - y).equals(0) for x, y in zip(a.bytes, b.bytes)]
    return (same[0] + same[1] + same[2] + same[3]).equals(4)


def select_word(condition, yes, no):
    return Word(tuple(select(condition, y, n) for y, n in zip(yes.bytes, no.bytes)))


# -- Bitwise operations and shifts ----------------------------------------------


def _bitwise(a, b, combine):
    a_bits = a if isinstance(a, list) else bits(a)
    b_bits = b if isinstance(b, list) else bits(b)
    return from_bits([combine(x, y) for x, y in zip(a_bits, b_bits)])


# Each bitwise result is a function of x + y in {0, 1, 2}: one small lookup.
def bitwise_and(a, b):
    return _bitwise(a, b, lambda x, y: (x + y).lookup((0, 0, 1)))


def bitwise_or(a, b):
    return _bitwise(a, b, lambda x, y: (x + y).lookup((0, 1, 1)))


def bitwise_xor(a, b):
    return _bitwise(a, b, lambda x, y: (x + y).lookup((0, 1, 0)))


def shift_amount(value):
    """The low five bits of a word, as bits (RISC-V ignores higher bits)."""
    low, _ = nibbles(value.bytes[0])
    high_bit = value.bytes[0].lookup(_BIT4)
    return [low.lookup(_NIBBLE_BITS[k]) for k in range(4)] + [high_bit]


def _barrel(values, amount, left, fill):
    for stage, bit in enumerate(amount):
        distance = 1 << stage
        shifted = []
        for i in range(32):
            source = i - distance if left else i + distance
            shifted.append(values[source] if 0 <= source < 32 else fill)
        values = [select(bit, s, v) for s, v in zip(shifted, values)]
    return values


# Bitwise operations and shifts accept a Word or its precomputed bit list, so an
# interpreter can share one bit decomposition among several functional units.
def shift_left(a, amount_bits):
    values = a if isinstance(a, list) else bits(a)
    return from_bits(_barrel(values, amount_bits, True, constant(0)))


def shift_right(a, amount_bits, arithmetic=False):
    values = a if isinstance(a, list) else bits(a)
    fill = values[31] if arithmetic else constant(0)
    return from_bits(_barrel(values, amount_bits, False, fill))


# -- Multiplication ---------------------------------------------------------------


def _byte_product(x, y_bits):
    """x * y for bytes, as (low byte, high byte), using y's bits."""
    total = constant(0)
    for k, bit in enumerate(y_bits):
        term = select(bit, x, 0)
        total = total + (term * (1 << k) if k else term)
    high = total.lookup(_PRODUCT_HIGH)
    return _low(total, high), high


def multiply(a, b, high_word=False):
    """Unsigned product: the low word, or with high_word=True the high word."""
    low, high = multiply_both(a, b, high_word)
    return high if high_word else low


def multiply_both(a, b, high_word=True, b_bits=None):
    """(low word, high word) of the unsigned product from one set of byte products.

    With high_word=False only the partial products reaching the low word are
    built, and the high word is None.
    """
    b_bits = bits(b) if b_bits is None else b_bits
    columns = [[] for _ in range(8)]
    for i, x in enumerate(a.bytes):
        for j in range(4):
            if not high_word and i + j > 3:
                continue
            low, high = _byte_product(x, b_bits[8 * j : 8 * j + 8])
            columns[i + j].append(low)
            if i + j + 1 < 8:
                columns[i + j + 1].append(high)
    out, c = [], constant(0)
    for column in columns[: 8 if high_word else 4]:
        total = c
        for term in column:
            total = total + term
        # At most 7 bytes plus a carry below 8: a lookup splits the column sum.
        c = total.lookup(_COLUMN_CARRY)
        out.append(_low(total, c))
    return Word(tuple(out[:4])), (Word(tuple(out[4:])) if high_word else None)


def mulh(a, b, signed_a=True, signed_b=True, high=None):
    """High word of a signed/unsigned product via the unsigned high word.

    For two's complement operands, hi(a*b) = hi_u(a*b) - [a<0]*b - [b<0]*a
    (mod 2**32), dropping the correction for operands treated as unsigned.
    `high` may supply an already computed unsigned high word.
    """
    result = multiply(a, b, high_word=True) if high is None else high
    if signed_a:
        result = sub(result, select_word(sign(a), b, word(0)))[0]
    if signed_b:
        result = sub(result, select_word(sign(b), a, word(0)))[0]
    return result


# -- Division ------------------------------------------------------------------------


def _double_plus(bytes_, bit):
    out, c = [], bit
    for byte in bytes_:
        low, c = carry(2 * byte + c)
        out.append(low)
    return out + [c]


def divide_unsigned(a, b):
    """(quotient, remainder) by restoring division; b = 0 gives (2**32-1, a)."""
    a_bits = bits(a)
    remainder = [constant(0)] * 4
    divisor = list(b.bytes) + [constant(0)]
    quotient = []
    for k in range(31, -1, -1):
        doubled = _double_plus(remainder, a_bits[k])  # Five bytes, < 2**33.
        difference, c = [], constant(1)
        for x, y in zip(doubled, divisor):
            low, c = carry(x + (255 - y) + c)
            difference.append(low)
        fits = c  # No borrow: doubled >= b.
        remainder = [select(fits, d, x) for d, x in zip(difference[:4], doubled[:4])]
        quotient.append(fits)
    return from_bits(quotient[::-1]), Word(tuple(remainder))


def divide_signed(a, b):
    """RISC-V DIV/REM: truncation toward zero; b = 0 and overflow per the spec."""
    negative_a, negative_b = sign(a), sign(b)
    magnitude_a = select_word(negative_a, negate(a), a)
    magnitude_b = select_word(negative_b, negate(b), b)
    quotient, remainder = divide_unsigned(magnitude_a, magnitude_b)
    zero_divisor = equal(b, word(0))
    # Quotient sign differs from the dividend's when signs differ, except b = 0.
    flip = select(zero_divisor, 0, (negative_a + negative_b).equals(1))
    quotient = select_word(flip, negate(quotient), quotient)
    quotient = select_word(zero_divisor, word(-1), quotient)
    remainder = select_word(negative_a, negate(remainder), remainder)
    # INT_MIN / -1 = INT_MIN with remainder 0 falls out of the unsigned path.
    return quotient, remainder


def is_word(value):
    return isinstance(value, Word) and all(isinstance(b, Expr) for b in value.bytes)
