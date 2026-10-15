"""Integer semantics, independent of transformer lowering and inference."""

from .ir import Expr, Machine, integer, nodes


def evaluate(expr: Expr, tokens: list[int]) -> int:
    """Evaluate every node once, in topological order, over one token prefix."""
    values: dict[Expr, int] = {}
    for node in nodes(expr):
        values[node] = _operation(node, tokens, values)
    return values[expr]


def _match(expr, tokens, values):
    """Newest tagged position whose keys (evaluated there) equal the query."""
    digits, period, phase, start, sentinel, end = expr.args[:6]
    query = [values[q] for q in expr.args[6 : 6 + digits]]
    keys = expr.args[6 + 2 * digits : 6 + 3 * digits]
    last = len(tokens) - 1 if end == -1 else min(len(tokens), end) - 1
    for j in range(last, start - 1, -1):
        if j % period == phase and evaluate_all(keys, tokens[: j + 1]) == query:
            return j
    if not start <= sentinel < len(tokens):
        raise ValueError("no sentinel in causal prefix")
    return sentinel


def evaluate_all(roots, tokens: list[int]) -> list[int]:
    """Evaluate several expressions over one prefix, sharing common nodes."""
    values: dict[Expr, int] = {}
    for root in roots:
        for node in nodes(root):
            if node not in values:
                values[node] = _operation(node, tokens, values)
    return [values[root] for root in roots]


def _operation(expr, tokens, values):
    op, args = expr.op, expr.args
    if op == "history":
        if len(tokens) <= args[0]:
            raise ValueError("insufficient history")
        return tokens[-1 - args[0]]
    if op == "constant":
        return args[0]
    if op == "position":
        if not tokens:
            raise ValueError("insufficient history")
        return len(tokens) - 1
    if op == "position_mod":
        if not tokens:
            raise ValueError("insufficient history")
        return (len(tokens) - 1 + args[1]) % args[0]
    if op in ("position_digit", "position_quotient"):
        if not tokens:
            raise ValueError("insufficient history")
        shifted = len(tokens) - 1 + args[1]
        if op == "position_digit":
            return shifted >> 8 * args[0] & 255 if shifted >= 0 else 0
        return shifted // args[0] if shifted >= 0 else -1
    if op == "match":
        return _match(expr, tokens, values)
    values = [values[a] if isinstance(a, Expr) else a for a in args]
    if op == "read_at":
        address = values[0]
        if not 0 <= address < len(tokens):
            raise ValueError("address outside causal prefix")
        return tokens[address]
    if op == "latest":
        _, period, phase, start = values
        tagged = [j for j in range(start, len(tokens)) if j % period == phase]
        if not tagged:
            raise ValueError("no tagged key in causal prefix")
        return max(tagged, key=lambda j: (-((tokens[j] - values[0]) ** 2), j))
    if op == "add":
        return values[0] + values[1]
    if op == "scale":
        return values[0] * values[1]
    if op == "relu":
        return max(0, values[0])
    if op == "clip":
        return min(values[2], max(values[1], values[0]))
    if op == "equals":
        return int(values[0] == values[1])
    if op == "mod":
        if values[0] < 0:
            raise ValueError("mod requires a nonnegative operand")
        return values[0] % values[1]
    if op == "lookup":
        x, lower, table = values
        if not 0 <= x - lower < len(table):
            raise ValueError("lookup argument outside its table")
        return table[x - lower]
    if op == "select":
        if values[0] not in (0, 1):
            raise ValueError("select condition must be 0 or 1")
        return values[1] if values[0] else values[2]
    raise ValueError(f"unknown operation: {op}")


def run(machine: Machine, prefix, max_steps=100):
    tokens = list(prefix)
    if integer(max_steps) < 0 or len(tokens) < machine.required_prefix:
        raise ValueError("invalid step budget or insufficient prefix")
    if any(not machine.lower <= integer(t) <= machine.upper for t in tokens):
        raise ValueError("token outside alphabet")
    for _ in range(max_steps):
        if tokens[-1] in machine.halt_values:
            break
        value = evaluate(machine.next_value, tokens)
        if not machine.lower <= value <= machine.upper:
            raise ValueError("transition outside alphabet")
        tokens.append(value)
    return tokens
