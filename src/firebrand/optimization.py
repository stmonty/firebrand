"""Opt-in, state-only assignment fusion before record/weight lowering."""

from dataclasses import replace
from types import MappingProxyType

from .ir import Expr, constant, nodes
from .sequence import Assign, Branch, Read, Write


def _bounds(expression, fields):
    """Prove field contracts before removing intermediate records.

    This deliberately conservative analysis also excludes physical trace access.
    It is local to this optimization; ordinary SequenceMachine is unchanged.
    """
    ranges = {}
    for node in nodes(expression):
        op, args = node.op, node.args
        if op == "state":
            field = fields[args[0]]
            bound = (field.lower, field.upper)
        elif op == "constant":
            bound = (args[0], args[0])
        elif op == "add":
            a, b = (ranges[x] for x in args)
            bound = (a[0] + b[0], a[1] + b[1])
        elif op == "scale":
            values = [v * args[1] for v in ranges[args[0]]]
            bound = (min(values), max(values))
        elif op == "relu":
            bound = tuple(max(0, v) for v in ranges[args[0]])
        elif op == "clip":
            bound = tuple(min(args[2], max(args[1], v)) for v in ranges[args[0]])
        elif op == "equals":
            bound = (0, 1)
        elif op == "mod":
            lo, hi = ranges[args[0]]
            if lo < 0:
                raise ValueError("fusion requires a nonnegative modulo operand")
            bound = (0, min(hi, args[1] - 1))
        elif op == "select":
            condition, yes, no = (ranges[x] for x in args)
            if condition[0] < 0 or condition[1] > 1:
                raise ValueError("fusion requires a Boolean select condition")
            bound = (min(yes[0], no[0]), max(yes[1], no[1]))
        else:
            raise ValueError(
                "assignment fusion requires state-only expressions; "
                f"physical trace access or unsupported operation: {op}"
            )
        ranges[node] = bound
    return ranges[expression]


def _substitute(expression, state, memo):
    if expression not in memo:
        if expression.op == "state":
            memo[expression] = state[expression.args[0]]
        else:
            memo[expression] = Expr(
                expression.op,
                tuple(_substitute(arg, state, memo) if isinstance(arg, Expr) else arg for arg in expression.args),
            )
    return memo[expression]


def _builder(expression):
    return lambda state: _substitute(expression, state, {})


def fuse_assignments(program):
    """Return a new SequenceMachine with adjacent Assign operations composed.

    Every label and non-Assign operation ends a group, including memory effects,
    explicit Halt, and the body boundary. No control-flow edge is crossed.
    Original callbacks execute once per occurrence during this pass. The returned
    program's builders substitute expressions at lowering, never at inference.

    All expressions must depend only on application fields/constants, including
    expressions outside fused groups. History/position reads are rejected because
    fusion changes the physical trace. Each original assignment must be provably
    within its declared field range; use explicit clipping when appropriate.

    Logical state at group boundaries, logical memory effects, and termination
    are preserved for valid programs. Intermediate records, PCs, physical memory
    links, and behavior at a fixed token budget intentionally change.
    """
    fields = {field.name: field for field in program.state.fields}
    symbols = MappingProxyType({name: Expr("state", (name,)) for name in fields})

    def build(value, *, destination=None, boolean=False):
        expression = constant(value(symbols) if callable(value) else value)
        lo, hi = _bounds(expression, fields)
        if destination is not None:
            field = fields[destination]
            if lo < field.lower or hi > field.upper:
                raise ValueError(
                    f"fusion cannot prove assignment to {destination!r} stays within its declared field range"
                )
        if boolean and (lo < 0 or hi > 1):
            raise ValueError("fusion requires a Boolean condition")
        return expression

    result, pending = [], {}
    current = dict(symbols)

    def flush():
        if pending:
            result.append(Assign({name: _builder(expr) for name, expr in pending.items()}))
            pending.clear()
        current.clear()
        current.update(symbols)

    for op in program.operations:
        if isinstance(op, Assign):
            # Evaluate all RHSs against the same pre-operation state (a swap
            # remains a swap), then expose them to the next Assign in the group.
            memo = {}
            updates = {
                name: _substitute(build(value, destination=name), current, memo) for name, value in op.updates.items()
            }
            current.update(updates)
            pending.update(updates)
            continue
        flush()
        if isinstance(op, Read):
            op = replace(op, address=_builder(build(op.address)))
        elif isinstance(op, Write):
            op = replace(op, address=_builder(build(op.address)), value=_builder(build(op.value)))
        elif isinstance(op, Branch):
            op = replace(op, condition=_builder(build(op.condition, boolean=True)))
        result.append(op)
    flush()
    return replace(program, operations=tuple(result), halt=_builder(build(program.halt, boolean=True)))
