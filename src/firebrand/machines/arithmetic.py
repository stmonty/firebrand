"""Exact multi-digit addition and multiplication as a compiled stream for a language model.

The stream reads tokens as small codes (digit, operator, "=", space, other).
When the text ends with `A × B =` or `A + B =`, it emits a space and then the
result, most significant digit first; everywhere else it emits PASS.
`reference_next` makes the same decision by direct parsing.
"""

from dataclasses import dataclass

from ..ir import Machine, constant, latest, position, read_at, select

PLUS, TIMES, EQUALS, SPACE, OTHER, PASS = 10, 11, 12, 13, 14, 15


def _is_digit(code):
    return (10 - code).clip(0, 1)


def _newer(a, b):
    """1 if position a > position b."""
    return (a - b).clip(0, 1)


@dataclass(frozen=True)
class ArithmeticSpec:
    digits: int = 12  # Longest operand.
    max_context: int = 4096

    def machine(self):
        n, p = self.digits, position()

        def find(code):
            at = latest(constant(code), period=1, phase=0, start=0)
            ok = read_at(at).equals(code)
            return select(ok, at, -1)  # -1: no such token yet.

        def valid(address):
            """1 if address is an existing position: 0 <= address <= p."""
            return select((address + 1).clip(0, 1), (p - address + 1).clip(0, 1), 0)

        def code_at(address):
            # Positions outside the prefix read position 0; callers ignore them.
            return read_at(select(valid(address), address, 0))

        def is_code(address, code):
            return select(valid(address), code_at(address).equals(code), 0)

        equals, plus, times = find(EQUALS), find(PLUS), find(TIMES)
        multiply = _newer(times, plus)
        op = select(multiply, times, plus)
        newest_space, newest_other = find(SPACE), find(OTHER)

        def operand(end):
            """Digits ending at `end`, least significant first: (digits, length, start)."""
            values, length, run = [], constant(0), constant(1)
            for i in range(n):
                address = end - i
                code = code_at(address)
                run = select(run, select(valid(address), _is_digit(code), 0), 0)
                values.append(select(run, code, 0).clip(0, 9))
                length = length + run
            return values, length, end - length + 1

        # B ends just before "=" (or a space before it); A just before the operator.
        b_end = equals - 1 - is_code(equals - 1, SPACE)
        b, b_length, b_start = operand(b_end)
        a_end = op - 1 - is_code(op - 1, SPACE)
        a, a_length, a_start = operand(a_end)
        # Between the operator and B: nothing, or one space. Before A: not a digit.
        joined = select(
            (b_start - 1 - op).equals(0), 1, select((b_start - 2 - op).equals(0), is_code(b_start - 1, SPACE), 0)
        )
        # (B's left edge is checked by `joined`, so B cannot be longer than `digits`.)
        a_bounded = 1 - select(valid(a_start - 1), _is_digit(code_at(a_start - 1)), 0)
        digits_out = [select(multiply, x, y) for x, y in zip(self._product(a, b), self._sum(a, b))]
        length = constant(1)
        for j in range(1, 2 * n):  # Highest nonzero digit, at least one digit.
            length = select(digits_out[j].equals(0), length, j + 1)
        # Answer progress: "=", then a space, then k digits so far.
        at_equals = (p - equals).equals(0)
        k = p - equals - 1
        spaced = is_code(equals + 1, SPACE)
        clean = select(_newer(equals + 2, newest_space), _newer(equals, newest_other), 0)
        unfinished = _newer(length, k)
        index = length - 1 - k
        digit = constant(0)
        for j in range(2 * n):
            digit = digit + select(index.equals(j), digits_out[j], 0)
        well_formed = select((equals + 1).clip(0, 1), select((op + 1).clip(0, 1), _newer(equals, op), 0), 0)
        well_formed = select(well_formed, select(a_length.clip(0, 1), select(b_length.clip(0, 1), joined, 0), 0), 0)
        well_formed = select(well_formed, a_bounded, 0)
        answering = select(spaced, select(clean, unfinished, 0), 0)
        active = select(well_formed, select(at_equals, 1, answering), 0)
        out = select(active, select(at_equals, SPACE, digit.clip(0, 9)), PASS)
        return Machine(out, 0, PASS, min_prefix=1, context_limit=self.max_context)

    def _product(self, a, b):
        """Digits of a × b (least significant first), from digit products and a carry chain."""
        n = self.digits
        table = tuple((x // 10) * (x % 10) for x in range(100))
        columns = [constant(0) for _ in range(2 * n)]
        for i in range(n):
            for j in range(n):
                columns[i + j] = columns[i + j] + (a[i] * 10 + b[j]).lookup(table)
        return self._carry(columns, 81 * n)

    def _sum(self, a, b):
        n = self.digits
        return self._carry([a[i] + b[i] for i in range(n)] + [constant(0)] * n, 18)

    def _carry(self, columns, column_max):
        """Base-10 digits of sum(columns[j] * 10**j), with an exact carry chain."""
        digits, carry, carry_max = [], constant(0), 0
        for column in columns:
            total = column + carry
            top = column_max + carry_max
            carry = total.lookup(tuple(v // 10 for v in range(top + 1)))
            digits.append(total - carry * 10)
            carry_max = top // 10
        return digits


def encode_text(tokens_text):
    """Codes for decoded token strings (see `code_of`)."""
    return [code_of(t) for t in tokens_text]


def code_of(text):
    """The code the arithmetic stream sees for one token's text."""
    stripped = text.strip()
    if len(text) == 1 and text in "0123456789":
        return int(text)
    if stripped in ("*", "×", "x", "times", "cdot"):  # LaTeX writes \times as "\" then "times".
        return TIMES
    if stripped == "+":
        return PLUS
    if stripped == "=":
        return EQUALS
    if text and (text.strip(" ") == "" or stripped == "\\"):  # A lone backslash starts a LaTeX command.
        return SPACE
    return OTHER


def reference_next(codes, digits=12):
    """What the stream emits after `codes`, by direct parsing (for tests)."""
    p = len(codes) - 1

    def newest(code):
        for i in range(p, -1, -1):
            if codes[i] == code:
                return i
        return -1

    equals, plus, times = newest(EQUALS), newest(PLUS), newest(TIMES)
    op = max(plus, times)
    if equals < 0 or op < 0 or op > equals:
        return PASS

    def operand(end):
        values = []
        while end >= 0 and codes[end] <= 9 and len(values) < digits:
            values.append(codes[end])
            end -= 1
        return values, end + 1

    b_end = equals - 1 - (equals >= 1 and codes[equals - 1] == SPACE)
    b, b_start = operand(b_end)
    a_end = op - 1 - (op >= 1 and codes[op - 1] == SPACE)
    a, a_start = operand(a_end)
    joined = b_start - 1 == op or (b_start - 2 == op and codes[b_start - 1] == SPACE)
    if not a or not b or not joined or (a_start >= 1 and codes[a_start - 1] <= 9):
        return PASS
    number = lambda ds: int("".join(str(d) for d in reversed(ds)))
    result = str(number(a) * number(b) if times > plus else number(a) + number(b))
    if p == equals:
        return SPACE
    k = p - equals - 1
    spaced = equals + 1 <= p and codes[equals + 1] == SPACE
    clean = all(c <= 9 for c in codes[equals + 2 : p + 1])
    if spaced and clean and k < len(result):
        return int(result[k])
    return PASS
