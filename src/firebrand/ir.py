"""Small integer expression IR. Construction has no global graph registry."""

from dataclasses import dataclass


def integer(value):
    if type(value) is not int:
        raise TypeError("expected a Python integer")
    return value


@dataclass(frozen=True, eq=False)
class Expr:
    op: str
    args: tuple = ()

    def __add__(self, other):
        return Expr("add", (self, constant(other)))

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -constant(other)

    def __rsub__(self, other):
        return constant(other) + -self

    def __mul__(self, coefficient):
        return Expr("scale", (self, integer(coefficient)))

    __rmul__ = __mul__

    def relu(self):
        return Expr("relu", (self,))

    def equals(self, value):
        """Integer equality with a constant, producing 0 or 1."""
        return Expr("equals", (self, integer(value)))

    def __mod__(self, divisor):
        if integer(divisor) <= 0:
            raise ValueError("modulus must be positive")
        return Expr("mod", (self, divisor))

    def lookup(self, table, lower=0):
        """table[x - lower]: any function of one bounded integer, as one FFN layer."""
        if type(table) is not tuple or not all(type(value) is int for value in table):
            table = tuple(integer(value) for value in table)
        if not table:
            raise ValueError("lookup table must be nonempty")
        return Expr("lookup", (self, integer(lower), table))

    def clip(self, lower, upper):
        integer(lower)
        integer(upper)
        if lower > upper:
            raise ValueError("clip lower exceeds upper")
        return Expr("clip", (self, lower, upper))


def constant(value):
    return value if isinstance(value, Expr) else Expr("constant", (integer(value),))


def history(offset=0):
    """Read the token offset positions before the last input token (0 = current)."""
    if integer(offset) < 0:
        raise ValueError("history offset must be nonnegative")
    return Expr("history", (offset,))


def read_at(address):
    """The token at a zero-based position computed by an expression; it must be in the prefix."""
    return Expr("read_at", (constant(address),))


def position():
    """Zero-based position of the most recent input token."""
    return Expr("position")


def position_mod(period, offset=0):
    """(position + offset) % period, as a positional input feature."""
    if integer(period) < 1 or integer(offset) < 0:
        raise ValueError("period must be positive and offset nonnegative")
    return Expr("position_mod", (period, offset))


def position_digit(power, offset=0):
    """Byte `power` of (position + offset), or 0 while it is negative. A positional input feature."""
    if not 0 <= integer(power) < 4:
        raise ValueError("digit power must be 0..3")
    return Expr("position_digit", (power, integer(offset)))


def position_quotient(period, offset=0):
    """floor((position + offset) / period), or -1 while it is negative. A positional input feature."""
    if integer(period) < 1:
        raise ValueError("period must be positive")
    return Expr("position_quotient", (period, integer(offset)))


def latest(key, *, period, phase, start):
    """Position of the newest tagged token whose value equals key (one attention head).

    Tagged positions j satisfy j >= start and j % period == phase. Without an
    exact match the result is the nearest value, so compare read_at(result) with key.
    """
    for value in (period, phase, start):
        integer(value)
    if period < 1 or not 0 <= phase < period or start < 0:
        raise ValueError("invalid latest-lookup tag")
    return Expr("latest", (constant(key), period, phase, start))


def match(query, keys, *, period, phase, start, sentinel, digit_limit=255, end=None):
    """Position of the newest tagged position whose key digits equal the query, else `sentinel`.

    `keys` are expressions evaluated at each candidate position j (for example
    history(2), history(1), history(0): a three-token key ending at j); `query`
    is evaluated at the current position. Tagged positions satisfy start <= j
    (< end, if given) and j % period == phase. The sentinel must be tagged; its
    key digits count as zero. Digits lie in 0..digit_limit.
    """
    query, keys = [constant(q) for q in query], [constant(k) for k in keys]
    for value in (period, phase, start, sentinel, digit_limit):
        integer(value)
    end = -1 if end is None else integer(end)
    if end != -1 and end <= sentinel:
        raise ValueError("the tag end must follow the sentinel")
    if not query or len(query) != len(keys):
        raise ValueError("query and keys need the same positive number of digits")
    if period < 1 or not 0 <= phase < period or start < 0 or digit_limit < 1:
        raise ValueError("invalid match tag or digit limit")
    if sentinel < start or sentinel % period != phase:
        raise ValueError("the sentinel must be a tagged position")
    squares = tuple(v * v for v in range(digit_limit + 1))
    at_sentinel = position().equals(sentinel)
    query = [q.clip(0, digit_limit) for q in query]
    keys = [select(at_sentinel, 0, k.clip(0, digit_limit)) for k in keys]
    return Expr(
        "match",
        (
            len(query),
            period,
            phase,
            start,
            sentinel,
            end,
            *query,
            *(q.lookup(squares) for q in query),
            *keys,
            *(k.lookup(squares) for k in keys),
        ),
    )


def select(condition, when_true, when_false):
    """Select a value using an integer 0/1 condition. Both branches are eager."""
    return Expr("select", (constant(condition), constant(when_true), constant(when_false)))


def nodes(root):
    """Topological order (iterative, so deep graphs do not hit the recursion limit)."""
    seen, ordered = {root}, []
    stack = [(root, iter([a for a in root.args if isinstance(a, Expr)]))]
    while stack:
        node, children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            ordered.append(node)
        elif child not in seen:
            seen.add(child)
            stack.append((child, iter([a for a in child.args if isinstance(a, Expr)])))
    return ordered


@dataclass(frozen=True)
class Machine:
    next_value: Expr
    lower: int
    upper: int
    halt_values: tuple[int, ...] = ()
    min_prefix: int = 1
    context_limit: int | None = None

    def __post_init__(self):
        if not isinstance(self.next_value, Expr):
            raise TypeError("next_value must be an expression")
        integer(self.lower)
        integer(self.upper)
        if self.lower > self.upper:
            raise ValueError("empty token alphabet")
        if integer(self.min_prefix) < 1:
            raise ValueError("min_prefix must be positive")
        if self.context_limit is not None and integer(self.context_limit) < self.min_prefix:
            raise ValueError("context_limit must cover min_prefix")
        for value in self.halt_values:
            if not self.lower <= integer(value) <= self.upper:
                raise ValueError("halt value outside alphabet")

    @property
    def required_prefix(self):
        return max(
            self.min_prefix, 1 + max((n.args[0] for n in nodes(self.next_value) if n.op == "history"), default=0)
        )
