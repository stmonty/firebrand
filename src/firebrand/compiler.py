"""Analytical lowering into named residual slots, 2D heads, and ReLU FFNs."""

from collections import defaultdict

import numpy as np

from .ir import Expr, Machine, integer, nodes
from .runtime import Sparse, Transformer

# Matrices with more entries than this are stored as coordinate lists.
DENSE_LIMIT = 1 << 20
SMALL = 1 << 16
MAX_NODES = 1 << 16
# Every node's range and every coefficient stay below MAGNITUDE, and positions
# below MAX_CONTEXT, so products, positional scores (up to 2 * MAX_CONTEXT**2),
# and lookup scores all remain exact float64 integers (positional scores stay
# below 2**51 at MAX_CONTEXT = 2**25).
MAX_CONTEXT = 1 << 25
MAGNITUDE = 1 << 26


class _Block:
    """One FFN node: hidden neurons reading residual rows, writing its slot."""

    def __init__(self, neurons):
        self.neurons = neurons
        self.inputs = []  # (residual row, neuron, weight)
        self.outputs = []  # (neuron, residual column, weight)

    def read(self, row, weights):
        self.inputs += [(row, col, w) for col, w in enumerate(weights) if w]

    def write(self, column, weights):
        self.outputs += [(col, column, w) for col, w in enumerate(weights) if w]


def _matrix(shape, entries, density=0.0):
    """Dense if small, else a coordinate list; duplicate entries are summed.

    With density > 0, a matrix above SMALL entries also needs that fraction of
    nonzeros to stay dense (large FFN stages are almost all zeros).
    """
    if not entries:
        rows = cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0)
    else:
        rows, cols, vals = (np.asarray(x) for x in zip(*entries))
    size = shape[0] * shape[1]
    if size <= DENSE_LIMIT and (size <= SMALL or len(entries) >= density * size):
        dense = np.zeros(shape)
        np.add.at(dense, (rows, cols), vals)
        return dense
    return Sparse(rows.astype(np.int64), cols.astype(np.int64), vals.astype(np.float64), shape)


def _ffn_stage(blocks, width):
    entries_in, entries_out, offset = [], [], 0
    for block in blocks:
        entries_in += [(row, offset + col, w) for row, col, w in block.inputs]
        entries_out += [(offset + col, column, w) for col, column, w in block.outputs]
        offset += block.neurons
    return (_matrix((width, offset), entries_in, density=0.01), _matrix((offset, width), entries_out, density=0.01))


def compile_machine(machine: Machine, *, max_context=None, pack=True) -> Transformer:
    """Lower an expression machine to weights.

    Each node goes in the earliest attention or FFN sublayer after its inputs.
    `pack=False` gives every node its own stage instead (easier to inspect; same outputs).
    """
    if max_context is None:
        max_context = machine.context_limit if machine.context_limit is not None else 4096
    if machine.context_limit is not None and integer(max_context) > machine.context_limit:
        raise ValueError("max_context exceeds the machine definition's context_limit")
    if not machine.required_prefix <= integer(max_context) <= MAX_CONTEXT:
        raise ValueError("max_context must cover the prefix and be at most 2**25")
    vocabulary = machine.upper - machine.lower + 1
    if vocabulary > 4096:
        raise ValueError("at most 4096 token values are supported")
    ordered = nodes(machine.next_value)
    if len(ordered) > MAX_NODES:
        raise ValueError(f"at most {MAX_NODES} expression nodes are supported")
    # Input columns: 0 constant one, 1 position, 2 position squared, 3 token value,
    # then positional features as nodes need them.
    features: list[dict] = []
    columns: dict = {}

    def feature(description):
        key = tuple(sorted(description.items()))
        if key not in columns:
            columns[key] = 4 + len(features)
            features.append(dict(description, column=columns[key]))
        return columns[key]

    for node in ordered:
        if node.op == "position_mod":
            feature({"kind": "mod", "period": node.args[0], "offset": node.args[1]})
        elif node.op == "position_digit":
            feature({"kind": "digit", "power": node.args[0], "offset": node.args[1]})
        elif node.op == "position_quotient":
            feature({"kind": "quotient", "period": node.args[0], "offset": node.args[1]})
        elif node.op == "latest":
            feature({"kind": "token_squared"})
            feature({"kind": "tag", "period": node.args[1], "phase": node.args[2], "start": node.args[3]})
        elif node.op == "match":
            period, phase, start = node.args[1:4]
            feature(_tag(period, phase, start, node.args[5]))
            feature({"kind": "point", "position": node.args[4]})
    base = 4 + len(features)
    slots = {node: base + i for i, node in enumerate(ordered)}
    width = base + len(ordered)
    ranges: dict[Expr, tuple[int, int]] = {}
    weights: dict = {}
    layout: list[dict] = []
    schedule: list[dict] = []
    embedding = np.zeros((vocabulary, base), dtype=np.float64)
    embedding[:, 0] = 1
    embedding[:, 3] = np.arange(machine.lower, machine.upper + 1)
    for description in features:
        if description["kind"] == "token_squared":
            embedding[:, description["column"]] = embedding[:, 3] ** 2
    if vocabulary * width <= DENSE_LIMIT:
        # Small models keep the full-width embedding; the extra columns are zero.
        embedding = np.pad(embedding, ((0, 0), (0, width - base)))
    weights["embedding"] = embedding
    heads = 0
    head_kinds: list[str] = []
    placement: dict = {}
    blocks: list = []
    match_heads: dict[str, dict] = {}
    for node in ordered:
        op, args, dest = node.op, node.args, slots[node]
        if op in ("history", "read_at"):
            bounds = (machine.lower, machine.upper)
            q, k = np.zeros((width, 2)), np.zeros((base, 2))
            q[0, 1] = 1
            if op == "history":
                q[1, 0], q[0, 0] = 1, -args[0]
            else:
                address_lo, address_hi = ranges[args[0]]
                if address_hi < 0 or address_lo >= max_context:
                    raise ValueError("address can never exist in compiled context")
                q[slots[args[0]], 0] = 1
            k[1, 0], k[2, 1] = 2, -1
            v, out = np.zeros((base, 1)), np.zeros((1, width))
            v[3, 0], out[0, dest] = 1, 1
            head_kinds.append("absolute")
        elif op == "latest":
            key, period, phase, start = args
            if start >= max_context:
                raise ValueError("latest-lookup tags can never exist in compiled context")
            # score = M * (2*key*a - a*a) + j + big * tag(j), with M > every j.
            # An exact match beats any other value by at least M, and big exceeds
            # every tagged/untagged score difference. All terms are exact integers.
            scale = max_context
            reach = max(abs(v) for v in ranges[key]) + max(abs(machine.lower), abs(machine.upper))
            big = scale * reach**2 + max_context
            if 2 * big >= 2**52:
                raise ValueError("latest-lookup scores exceed exact float64 integers")
            q, k = np.zeros((width, 4)), np.zeros((base, 4))
            q[slots[key], 0] = 2 * scale
            q[0, 1:] = [-scale, 1, big]
            k[3, 0] = 1
            k[feature({"kind": "token_squared"}), 1] = 1
            k[1, 2] = 1
            k[feature({"kind": "tag", "period": period, "phase": phase, "start": start}), 3] = 1
            v, out = np.zeros((base, 1)), np.zeros((1, width))
            v[1, 0], out[0, dest] = 1, 1
            bounds = (start, max_context - 1)
            head_kinds.append("latest")
        elif op == "match":
            digits, period, phase, start, sentinel, end = args[:6]
            groups = [args[6 + i * digits : 6 + (i + 1) * digits] for i in range(4)]
            query, query_squares, keys, key_squares = groups
            if sentinel >= max_context:
                raise ValueError("match sentinel can never exist in compiled context")
            # score = M*sum(2*q*k - k*k) + j + big*tag(j) + [j = sentinel]*(M*sum(q*q) - M/2).
            # M > 2*max_context: a match beats every non-match by M, the sentinel
            # sits halfway between, and big lifts every tagged position above every
            # untagged one. Keys come from computed slots at each candidate position.
            scale = 2 * max_context + 2
            limit = max(max(abs(v) for v in ranges[n]) for n in (*query, *keys))
            spread = scale * digits * limit * limit
            big = 3 * spread + 2 * max_context + scale
            if 4 * big >= 2**52:
                raise ValueError("match scores exceed exact float64 integers")
            dims = 2 * digits + 3
            q, k = np.zeros((width, dims)), np.zeros((width, dims))
            for i in range(digits):
                q[slots[query[i]], i] = 2 * scale
                q[0, digits + i] = -scale
                q[slots[query_squares[i]], 2 * digits] += scale
                k[slots[keys[i]], i] = 1
                k[slots[key_squares[i]], digits + i] = 1
            q[0, 2 * digits :] = [-scale // 2, 1, big]
            k[feature({"kind": "point", "position": sentinel}), 2 * digits] = 1
            k[1, 2 * digits + 1] = 1
            k[feature(_tag(period, phase, start, end)), 2 * digits + 2] = 1
            v, out = np.zeros((base, 1)), np.zeros((1, width))
            v[1, 0], out[0, dest] = 1, 1
            bounds = (start, max_context - 1)
            head_kinds.append("match")
            match_heads[str(heads)] = dict(
                digits=digits, period=period, phase=phase, start=start, sentinel=sentinel, scale=scale
            )
            if end != -1:
                match_heads[str(heads)]["end"] = end
            if width * dims > DENSE_LIMIT:
                rows, cols = np.nonzero(k)
                k = Sparse(rows, cols, k[rows, cols], k.shape)  # type: ignore[assignment]
        else:
            bounds, block = _ffn_node(node, ranges, slots, dest, max_context, feature)
        if op in ("history", "read_at", "latest", "match"):
            if width * base <= DENSE_LIMIT:
                v = np.pad(v, ((0, width - base), (0, 0)))
                if op != "match":
                    k = np.pad(k, ((0, width - base), (0, 0)))
            for name, array in (("q", q), ("k", k), ("v", v), ("o", out)):
                weights[f"head_{heads}_{name}"] = array
            stage = f"head_{heads}"
            schedule.append({"kind": "attention", "index": heads})
            placement[node] = ("attention", heads)
            heads += 1
        else:
            index = len(blocks)
            blocks.append(block)
            stage = f"ff_{index}"
            schedule.append({"kind": "ffn", "index": index})
            placement[node] = ("ffn", index)
        # Deliberately conservative integer range and coefficient contract.
        literals = [a for a in args if type(a) is int]
        if any(abs(v) > MAGNITUDE for v in (*bounds, *literals)):
            raise ValueError("integer range or coefficient exceeds 2**26")
        ranges[node] = bounds
        layout.append({"slot": dest, "operation": op, "range": list(bounds), "stage": stage})
    lo, hi = ranges[machine.next_value]
    if lo < machine.lower or hi > machine.upper:
        raise ValueError("inferred output range exceeds alphabet; use clip or widen the alphabet")
    values = np.arange(machine.lower, machine.upper + 1, dtype=np.float64)
    # Argmax_v (2*v*x - v*v) selects the integer x with a unit margin.
    output = slots[machine.next_value]
    weights["unembedding"] = _matrix(
        (width, vocabulary),
        [*((output, i, 2 * v) for i, v in enumerate(values)), *((0, i, -v * v) for i, v in enumerate(values) if v)],
    )
    if pack:
        layers, schedule = _pack(ordered, placement, blocks, weights, layout, width)
    else:
        layers = len(blocks)
        for i, block in enumerate(blocks):
            weights[f"ff_{i}_in"], weights[f"ff_{i}_out"] = _ffn_stage([block], width)
    config = dict(
        format_version=2,
        lower=machine.lower,
        upper=machine.upper,
        halt_values=list(machine.halt_values),
        required_prefix=machine.required_prefix,
        max_context=max_context,
        heads=heads,
        layers=layers,
        width=width,
        attention="hardmax",
        dtype="float64",
        layout=layout,
        schedule=schedule,
        addressing="absolute_checked",
    )
    if features or pack:
        config.update(format_version=3, position_features=features, head_kinds=head_kinds)
    if match_heads:
        config["match_heads"] = match_heads
    if any(isinstance(w, Sparse) for w in weights.values()) or embedding.shape[1] < width:
        config.update(format_version=4, input_width=base)
    return Transformer(config, weights)


def _tag(period, phase, start, end=-1):
    """A tag feature description."""
    description = {"kind": "tag", "period": period, "phase": phase, "start": start}
    if end != -1:
        description["end"] = end
    return description


def _ffn_node(node, ranges, slots, dest, max_context, feature):
    """Bounds and a ReLU block computing one non-attention node exactly."""
    op, args = node.op, node.args
    linear = defaultdict(int)
    if op == "constant":
        bounds = (args[0], args[0])
        linear[0] = args[0]
    elif op == "position":
        bounds = (0, max_context - 1)
        linear[1] = 1
    elif op == "position_mod":
        bounds = (0, args[0] - 1)
        linear[feature({"kind": "mod", "period": args[0], "offset": args[1]})] = 1
    elif op == "position_digit":
        bounds = (0, 255)
        linear[feature({"kind": "digit", "power": args[0], "offset": args[1]})] = 1
    elif op == "position_quotient":
        bounds = (-1, max(-1, (max_context - 1 + args[1]) // args[0]))
        linear[feature({"kind": "quotient", "period": args[0], "offset": args[1]})] = 1
    elif op == "add":
        a, b = (ranges[arg] for arg in args)
        bounds = (a[0] + b[0], a[1] + b[1])
        for arg in args:
            linear[slots[arg]] += 1
    elif op == "scale":
        products = [v * args[1] for v in ranges[args[0]]]
        bounds = (min(products), max(products))
        linear[slots[args[0]]] = args[1]
    elif op == "relu":
        bounds = tuple(max(0, v) for v in ranges[args[0]])
        linear[slots[args[0]]] = 1
    elif op == "clip":
        bounds = tuple(min(args[2], max(args[1], v)) for v in ranges[args[0]])
    elif op == "equals":
        bounds = (0, 1)
    elif op == "mod":
        lower, upper = ranges[args[0]]
        if lower < 0:
            raise ValueError("mod requires a nonnegative inferred operand range")
        bounds = (0, min(upper, args[1] - 1))
        if upper // args[1] > 4096:
            raise ValueError("mod construction exceeds 4096 threshold pairs")
    elif op == "lookup":
        x, lower, table = args
        lo, hi = ranges[x]
        if lo < lower or hi >= lower + len(table):
            raise ValueError("lookup argument range exceeds its table")
        reachable = table[lo - lower : hi - lower + 1]
        bounds = (min(reachable), max(reachable))
    elif op == "select":
        if ranges[args[0]][0] < 0 or ranges[args[0]][1] > 1:
            raise ValueError("select condition must have range 0..1")
        bounds = (min(ranges[args[1]][0], ranges[args[2]][0]), max(ranges[args[1]][1], ranges[args[2]][1]))
    else:
        raise ValueError(f"unsupported operation: {op}")

    if op == "equals":
        block = _Block(3)
        block.read(slots[args[0]], [1, 1, 1])
        block.read(0, [1 - args[1], -args[1], -1 - args[1]])
        block.write(dest, [1, -2, 1])
    elif op == "mod":
        # x % d = x - d * sum_k 1[x >= k*d], on bounded integers.
        count = ranges[args[0]][1] // args[1]
        block = _Block(1 + 2 * count)
        block.read(slots[args[0]], [1] * (1 + 2 * count))
        constants, weights = [0], [1]
        for k in range(1, count + 1):
            constants += [1 - k * args[1], -k * args[1]]
            weights += [-args[1], args[1]]
        block.read(0, constants)
        block.write(dest, weights)
    elif op == "lookup":
        # f(x) = f(lo) + sum over change points v of (f(v) - f(v-1)) * 1[x >= v],
        # with 1[x >= v] = ReLU(x - v + 1) - ReLU(x - v) on integers. The first
        # neuron is ReLU(1), which carries the constant f(lo).
        x, lower, table = args
        lo, hi = ranges[x]
        steps = [
            (v, table[v - lower] - table[v - 1 - lower])
            for v in range(lo + 1, hi + 1)
            if table[v - lower] != table[v - 1 - lower]
        ]
        block = _Block(1 + 2 * len(steps))
        block.read(slots[x], [0] + [1] * (2 * len(steps)))
        constants, weights = [1], [table[lo - lower]]
        for v, delta in steps:
            constants += [1 - v, -v]
            weights += [delta, -delta]
        block.read(0, constants)
        block.write(dest, weights)
    elif op == "select":
        block = _Block(4)
        condition, yes, no = args
        yes_bound = max(abs(v) for v in ranges[yes])
        no_bound = max(abs(v) for v in ranges[no])
        block.read(slots[yes], [1, -1, 0, 0])
        block.read(slots[no], [0, 0, 1, -1])
        block.read(slots[condition], [yes_bound, yes_bound, -no_bound, -no_bound])
        block.read(0, [-yes_bound, -yes_bound, 0, 0])
        block.write(dest, [1, -1, 1, -1])
    elif op == "clip":
        # lo + ReLU(x-lo) - ReLU(x-hi), including signed lo.
        block = _Block(3)
        block.read(slots[args[0]], [1, 1, 0])
        block.read(0, [-args[1], -args[2], 1])
        block.write(dest, [1, -1, args[1]])
    elif op == "relu":
        block = _Block(1)
        for row, w in linear.items():
            block.read(row, [w])
        block.write(dest, [1])
    else:
        # Signed linear expressions are ReLU(z) - ReLU(-z).
        block = _Block(2)
        for row, w in linear.items():
            block.read(row, [w, -w])
        block.write(dest, [1, -1])
    return bounds, block


def _pack(ordered, placement, blocks, weights, layout, width):
    """Group nodes into alternating attention/FFN sublayers by dependency depth.

    Level 2b is block b's attention sublayer and 2b+1 its FFN sublayer. A node
    goes in the earliest sublayer of its kind after all of its inputs.
    """
    level = {}
    for node in ordered:
        ready = max((level[a] for a in node.args if isinstance(a, Expr)), default=-1)
        kind = placement[node][0]
        candidate = ready + 1
        if (candidate % 2 == 0) != (kind == "attention"):
            candidate += 1
        level[node] = candidate
    groups = {}
    for node in ordered:
        groups.setdefault(level[node], []).append(node)
    schedule, names, layers = [], {}, 0
    for value in sorted(groups):
        members = groups[value]
        if value % 2 == 0:
            heads = [placement[node][1] for node in members]
            name = f"attention_{len(schedule) - layers}"
            schedule.append({"kind": "attention", "heads": heads, "name": name})
        else:
            stage = [blocks[placement[node][1]] for node in members]
            weights[f"ff_{layers}_in"], weights[f"ff_{layers}_out"] = _ffn_stage(stage, width)
            name = f"ff_{layers}"
            schedule.append({"kind": "ffn", "index": layers, "name": name})
            layers += 1
        for node in members:
            names[node] = name
    for node, entry in zip(ordered, layout):
        entry["stage"] = names[node]
    return layers, schedule
