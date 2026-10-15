"""Compile restricted Python functions into transformer weights.

    @program(max_context=1 << 14)
    def gcd(a: Int[0, 255], b: Int[0, 255]):
        while a != 0 and b != 0:
            if a > b:
                a = a - b
            else:
                b = b - a
        return a + b

    gcd(12, 8)                 # 4, by running the function as ordinary Python
    gcd.run(a=12, b=8).result  # 4, from the compiled transformer

The function's source is parsed with `ast` and lowered to a SequenceMachine:
each statement becomes a few record-sized steps. Every variable is a bounded
integer: parameters and locals declare their range with Int[lo, hi]; `for`
loops over range() infer theirs. Array[n, lo, hi] parameters are read-only data
in the prompt, read by attention without a step. Memory(n, lo, hi) arrays are
writable (newest write wins, one step per access). emit(x) appends to the
output; return x halts with a result. An assignment that may leave its
variable's range is clipped; running the function as Python is the reference.
"""

import ast
import inspect
import itertools
import textwrap
from dataclasses import dataclass, field

from . import compile_machine
from .ir import constant, read_at, select
from .memory import IndexedMemory
from .records import Field, Record
from .sequence import Assign, Branch, Halt, Jump, Label, Read, SequenceMachine, Write

TABLE_LIMIT = 4096  # Largest lookup table built for *, //, and % of variables.


# -- Annotations and Python-side behaviour ----------------------------------------


@dataclass(frozen=True)
class IntType:
    lower: int
    upper: int


@dataclass(frozen=True)
class ArrayType:
    length: int
    lower: int
    upper: int


class Int:
    """Int[lo, hi]: an integer variable in lo..hi."""

    def __class_getitem__(cls, bounds):
        lower, upper = bounds
        if lower > upper:
            raise ValueError("Int[lo, hi] needs lo <= hi")
        return IntType(lower, upper)


class Array:
    """Array[n, lo, hi]: a read-only input array of n integers in lo..hi."""

    def __class_getitem__(cls, spec):
        length, lower, upper = spec
        if length < 1 or lower > upper:
            raise ValueError("Array[n, lo, hi] needs n >= 1 and lo <= hi")
        return ArrayType(length, lower, upper)


class Memory(list):
    """Memory(n, lo, hi): a writable array of n integers in lo..hi, initially zero."""

    def __init__(self, length, lower, upper):
        super().__init__([0] * length)
        self.lower, self.upper = lower, upper


_outputs = None


def emit(value):
    """Append a value to the program's output."""
    if _outputs is not None:
        _outputs.append(int(value))


def clip(value, lower, upper):
    return min(upper, max(lower, value))


# -- Program objects ---------------------------------------------------------


@dataclass
class Run:
    result: int | None
    outputs: list
    steps: int
    tokens: int
    halted: bool


def program(function=None, *, max_context=1 << 14):
    """Decorator: compile a restricted Python function into a transformer."""
    if function is None:
        return lambda f: Program(f, max_context)
    return Program(function, max_context)


class Program:
    def __init__(self, function, max_context):
        self.function, self.max_context = function, max_context
        self.name = function.__name__
        self.__doc__ = function.__doc__
        self._lowered = _Lowering(function, max_context)
        self._model = None

    # Python semantics: the reference.
    def __call__(self, *args, **kwargs):
        return self.reference(*args, **kwargs)[0]

    def reference(self, *args, **kwargs):
        """(result, outputs) of running the function as ordinary Python."""
        global _outputs
        saved, _outputs = _outputs, []
        try:
            result = self.function(*args, **kwargs)
            return (None if result is None else int(result)), list(_outputs)
        finally:
            _outputs = saved

    # Compilation.
    @property
    def machine(self):
        """The SequenceMachine this function lowers to."""
        return self._lowered.machine

    @property
    def operations(self):
        return self._lowered.machine.operations

    @property
    def model(self):
        if self._model is None:
            self._model = compile_machine(self.machine.lower_machine())
        return self._model

    def prompt(self, *args, **kwargs):
        bound = inspect.signature(self.function).bind(*args, **kwargs)
        return self._lowered.prompt(bound.arguments)

    def run(self, *args, backend="c", max_steps=None, **kwargs):
        """Run the compiled transformer: its result, outputs, and cost."""
        prompt = self.prompt(*args, **kwargs)
        protocol = self.machine.protocol
        room = (protocol.max_context - protocol.prefix_length) // protocol.record.width
        steps = room if max_steps is None else min(room, max_steps)
        trace = self.model.generate(prompt, protocol.token_budget(steps), backend=backend)
        return self._lowered.decode(trace)

    def listing(self):
        """The operations this function lowers to, one step each, with labels."""
        return self._lowered.listing()

    def __repr__(self):
        return f"<program {self.name}: {len(self.machine.executable)} operations>"


# -- Lowering -----------------------------------------------------------------


@dataclass
class _Variable:
    kind: str  # "field", "input", or "memory"
    lower: int
    upper: int
    length: int = 0  # Arrays.
    offset: int = 0  # Arrays: payload or memory offset.


@dataclass
class _Lowering:
    function: object
    max_context: int
    ops: list = field(default_factory=list)

    def __post_init__(self):
        try:
            source = textwrap.dedent(inspect.getsource(self.function))
        except OSError as error:
            raise TypeError(
                f"@program reads {self.function.__name__}'s source, which Python cannot find: "
                "define it in a file or a notebook cell"
            ) from error
        tree = self.tree = ast.parse(source).body[0]
        if not isinstance(tree, ast.FunctionDef):
            raise TypeError("@program needs a function definition")
        self.globals = dict(self.function.__globals__)
        if self.function.__closure__:
            self.globals.update(
                zip(self.function.__code__.co_freevars, (c.cell_contents for c in self.function.__closure__))
            )
        self.variables, self.fields, self.parameters = {}, [], []
        self.payload, self.memory_cells, self.memory_limit = 0, 0, 0
        self.labels, self.temps, self.loops, self.emits = 0, 0, [], set()
        self.result_range, self.output_range = None, None
        self.text = {}  # id(operation) -> source, for listings.
        for argument in tree.args.args:
            self._parameter(argument)
        self._declare_memory(tree.body)
        self._block(tree.body)
        self.ops.append(Halt())
        self._assemble()

    # Declarations.
    def _annotation(self, node):
        try:
            return eval(compile(ast.fix_missing_locations(ast.Expression(node)), "<annotation>", "eval"), self.globals)
        except Exception as error:
            raise SyntaxError(f"cannot read annotation {ast.unparse(node)}: {error}") from None

    def _parameter(self, argument):
        if argument.annotation is None:
            raise SyntaxError(f"parameter {argument.arg} needs a type: Int[lo, hi] or Array[n, lo, hi]")
        kind = self._annotation(argument.annotation)
        if isinstance(kind, IntType):
            self._field(argument.arg, kind.lower, kind.upper)
        elif isinstance(kind, ArrayType):
            self.variables[argument.arg] = _Variable("input", kind.lower, kind.upper, kind.length, self.payload)
            self.payload += kind.length
        else:
            raise SyntaxError(f"parameter {argument.arg}: expected Int[lo, hi] or Array[n, lo, hi]")
        self.parameters.append(argument.arg)

    def _field(self, name, lower, upper):
        if name in self.variables:
            raise SyntaxError(f"{name} is declared twice")
        self.variables[name] = _Variable("field", lower, upper)
        self.fields.append(name)

    def _declare_memory(self, body):
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and self.globals.get(node.value.func.id) is Memory
            ):
                length, lower, upper = (self._constant(a) for a in node.value.args)
                if lower < 0:
                    raise SyntaxError("Memory values must be nonnegative")
                name = node.targets[0].id
                self.variables[name] = _Variable("memory", lower, upper, length, self.memory_cells)
                self.memory_cells += length
                self.memory_limit = max(self.memory_limit, upper)

    def _constant(self, node):
        value = self._fold(node)
        if value is None:
            raise SyntaxError(f"expected a constant: {ast.unparse(node)}")
        return value

    def _fold(self, node):
        """The value of a constant expression (literals and global ints), or None."""
        if isinstance(node, ast.Constant):
            return int(node.value) if isinstance(node.value, (int, bool)) else None
        if any(isinstance(n, ast.Name) and n.id in self.variables for n in ast.walk(node)):
            return None
        try:
            expression = ast.fix_missing_locations(ast.Expression(node))
            value = eval(
                compile(expression, "<constant>", "eval"),
                {"__builtins__": {}},
                {k: v for k, v in self.globals.items() if isinstance(v, int)},
            )
        except Exception:
            return None
        return int(value) if isinstance(value, int) else None

    # Statements.
    def _label(self, hint):
        self.labels += 1
        return f"{hint}_{self.labels}"

    def _block(self, statements):
        for statement in statements:
            self._statement(statement)

    def _statement(self, node):
        self.temps = 0
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and self.globals.get(node.value.func.id) is Memory
        ):
            return  # Declared up front.
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                raise SyntaxError("annotated assignments need a plain name")
            kind = self._annotation(node.annotation)
            if not isinstance(kind, IntType):
                raise SyntaxError(f"{node.target.id}: local variables are Int[lo, hi]")
            self._field(node.target.id, kind.lower, kind.upper)
            if node.value is not None:
                self._assign(node.target.id, node.value)
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise SyntaxError("assign one target at a time")
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                self._store(target, node.value)
            elif isinstance(target, ast.Tuple):
                self._parallel(target, node.value)
            elif isinstance(target, ast.Name):
                self._assign(target.id, node.value)
            else:
                raise SyntaxError(f"cannot assign to {ast.unparse(target)}")
        elif isinstance(node, ast.AugAssign):
            value = ast.BinOp(left=_load(node.target), op=node.op, right=node.value)
            if isinstance(node.target, ast.Subscript):
                self._store(node.target, value)
            else:
                self._assign(node.target.id, value)
        elif isinstance(node, ast.If):
            self._if(node)
        elif isinstance(node, ast.While):
            self._while(node)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, ast.Return):
            if node.value is not None:
                reads, value = self._hoist(node.value)
                self.ops += reads
                self.result_range = _union(self.result_range, self._range(value))
                self._add(Assign({"_result": self._builder(value, "_result")}), f"return {ast.unparse(value)}")
            self.ops.append(Halt())
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and self.globals.get(node.value.func.id) is emit
        ):
            if len(node.value.args) != 1:
                raise SyntaxError("emit takes one value")
            reads, value = self._hoist(node.value.args[0])
            self.ops += reads
            self.output_range = _union(self.output_range, self._range(value))
            op = Assign({"_out": self._builder(value, "_out")})
            self.emits.add(id(op))
            self._add(op, f"emit({ast.unparse(value)})")
        elif isinstance(node, (ast.Break, ast.Continue)):
            if not self.loops:
                raise SyntaxError("break or continue outside a loop")
            self.ops.append(Jump(self.loops[-1][1 if isinstance(node, ast.Break) else 0]))
        elif isinstance(node, ast.Pass) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            pass  # Docstrings and pass.
        else:
            raise SyntaxError(f"unsupported statement: {ast.unparse(node).splitlines()[0]}")

    def _assign(self, name, value):
        if name not in self.variables:
            raise SyntaxError(f"{name} needs a declaration: {name}: Int[lo, hi] = ...")
        if self.variables[name].kind != "field":
            raise SyntaxError(f"cannot assign to array {name}")
        reads, value = self._hoist(value)
        self._range(value)  # Checks the expression now, not at compile time.
        self.ops += reads
        self._add(Assign({name: self._builder(value, name)}), f"{name} = {ast.unparse(value)}")

    def _parallel(self, targets, values):
        """a, b = x, y: simultaneous assignment in one step."""
        if not isinstance(values, ast.Tuple) or len(values.elts) != len(targets.elts):
            raise SyntaxError("tuple assignment needs a tuple of the same length")
        updates, sources = {}, []
        for target, value in zip(targets.elts, values.elts):
            if not isinstance(target, ast.Name) or target.id not in self.variables:
                raise SyntaxError("tuple assignment needs declared scalar variables")
            reads, value = self._hoist(value)
            self._range(value)
            self.ops += reads
            updates[target.id] = self._builder(value, target.id)
            sources.append((target.id, ast.unparse(value)))
        self._add(Assign(updates), ", ".join(t for t, _ in sources) + " = " + ", ".join(v for _, v in sources))

    def _store(self, target, value):
        name = target.value.id if isinstance(target.value, ast.Name) else None
        variable = self.variables.get(name)
        if variable is None or variable.kind != "memory":
            raise SyntaxError(f"only Memory arrays can be written: {ast.unparse(target)}")
        index_reads, index = self._hoist(target.slice)
        value_reads, value = self._hoist(value)
        self.ops += index_reads + value_reads
        lower, upper = self._range(value)
        clipped = lower < variable.lower or upper > variable.upper
        index_builder, value_builder = self._expression(index), self._expression(value)

        def address(state, offset=variable.offset, length=variable.length):
            return (index_builder(state).clip(0, length - 1)) + offset

        def stored(state):
            built = value_builder(state)
            return built.clip(variable.lower, variable.upper) if clipped else built

        self._add(Write(address, stored), f"{name}[{ast.unparse(index)}] = {ast.unparse(value)}")

    def _if(self, node):
        then_label, else_label, end_label = (self._label(k) for k in ("then", "else", "endif"))
        self._branch(node.test, then_label, else_label)
        self.ops.append(Label(then_label))
        self._block(node.body)
        self.ops.append(Jump(end_label))
        self.ops.append(Label(else_label))
        self._block(node.orelse)
        self.ops.append(Label(end_label))

    def _branch(self, test, when_true, when_false):
        reads, test = self._hoist(test)
        self._range(test)
        self.ops += reads
        condition = self._expression(test, truth=True)
        self._add(Branch(condition, when_true, when_false), f"if {ast.unparse(test)}")

    def _while(self, node):
        """Tested on entry and again at the bottom of each iteration (no jump back)."""
        if node.orelse:
            raise SyntaxError("while ... else is not supported")
        body, cont, end = (self._label(k) for k in ("while", "next", "endwhile"))
        self._branch(node.test, body, end)
        self.ops.append(Label(body))
        self.loops.append((cont, end))
        self._block(node.body)
        self.loops.pop()
        self.ops.append(Label(cont))
        self._branch(node.test, body, end)
        self.ops.append(Label(end))

    def _for(self, node):
        """for i in range(start, stop[, step]) with a constant step."""
        call = node.iter
        if (
            not (
                isinstance(node.target, ast.Name)
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "range"
                and 1 <= len(call.args) <= 3
            )
            or node.orelse
        ):
            raise SyntaxError("for loops must be: for name in range(...)")
        args = call.args
        start, stop = (ast.Constant(0), args[0]) if len(args) == 1 else (args[0], args[1])
        step = self._constant(args[2]) if len(args) == 3 else 1
        if step == 0:
            raise SyntaxError("range step must not be zero")
        start_range, stop_range = self._range(start), self._range(stop)
        name = node.target.id
        low, high = min(start_range[0], stop_range[0]), max(start_range[1], stop_range[1])
        # The loop variable counts directly unless the program could tell the
        # difference: Python leaves it at the last value, the counter one step on.
        direct = name not in self.variables and self._local_to(name, node)
        counter = name if direct else f"_for{self.labels}_{name}"
        self._field(counter, low - abs(step), high + abs(step))
        if not direct and name not in self.variables:
            self._field(name, low, high)
        if self._fold(stop) is not None:  # A constant bound needs no field.
            limit_node = ast.Constant(self._fold(stop))
            self._assign(counter, start)
        else:
            limit = f"_stop{self.labels}_{name}"
            self._field(limit, *stop_range)
            limit_node = ast.Name(limit)
            self._parallel(ast.Tuple(elts=[ast.Name(counter), ast.Name(limit)]), ast.Tuple(elts=[start, stop]))
        body, cont, end = (self._label(k) for k in ("for", "next", "endfor"))
        test = ast.Compare(left=ast.Name(counter), ops=[ast.Lt() if step > 0 else ast.Gt()], comparators=[limit_node])
        self._branch(test, body, end)
        self.ops.append(Label(body))
        if not direct:
            self._assign(name, ast.Name(counter))
        self.loops.append((cont, end))
        self._block(node.body)
        self.loops.pop()
        self.ops.append(Label(cont))
        self._assign(counter, ast.BinOp(left=ast.Name(counter), op=ast.Add(), right=ast.Constant(step)))
        self._branch(test, body, end)
        self.ops.append(Label(end))

    def _local_to(self, name, loop):
        """Whether `name` appears only in this loop, and is never assigned in its body."""
        inside = {id(n) for n in ast.walk(loop)}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Name) and n.id == name and id(n) not in inside:
                return False
        return not any(
            isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
            for statement in loop.body
            for n in ast.walk(statement)
        )

    def _add(self, op, text):
        self.text[id(op)] = text
        self.ops.append(op)

    def listing(self):
        lines, index = [], 0
        for op in self.machine.operations:
            if isinstance(op, Label):
                lines.append(f"{op.name}:")
                continue
            if isinstance(op, Jump):
                text = f"goto {op.target}"
            elif isinstance(op, Halt):
                text = "halt"
            elif isinstance(op, Branch):
                text = f"{self.text[id(op)]}  -> {op.when_true} else {op.when_false}"
            else:
                text = self.text[id(op)]
            lines.append(f"  {index:3}  {text}")
            index += 1
        return "\n".join(lines)

    # Memory reads become steps before the statement that uses them.
    def _hoist(self, node):
        reads = []
        lowering = self

        class Rewrite(ast.NodeTransformer):
            def visit_Subscript(self, sub):
                self.generic_visit(sub)
                name = sub.value.id if isinstance(sub.value, ast.Name) else None
                variable = lowering.variables.get(name)
                if variable is None or variable.kind != "memory":
                    return sub
                temp = lowering._temp(variable)
                index = lowering._expression(sub.slice)
                read = Read(temp, lambda state, index=index, v=variable: index(state).clip(0, v.length - 1) + v.offset)
                lowering.text[id(read)] = f"{temp} = {ast.unparse(sub)}"
                reads.append(read)
                return ast.Name(temp)

        rewritten = Rewrite().visit(ast.parse(ast.unparse(node), mode="eval").body)
        return reads, rewritten

    def _temp(self, variable):
        """A field for a memory read. Reads feed the very next steps, so each
        statement reuses the same few temporaries."""
        name = f"_read{self.temps}"
        self.temps += 1
        if name not in self.variables:
            self.variables[name] = _Variable("field", 0, self.memory_limit)
            self.fields.append(name)
        return name

    # Expressions: interval analysis and builders.
    def _range(self, node):
        folded = self._fold(node)
        if folded is not None:
            return folded, folded
        if isinstance(node, ast.Name):
            variable = self.variables.get(node.id)
            if variable is None or variable.kind != "field":
                raise SyntaxError(f"unknown variable {node.id}")
            return variable.lower, variable.upper
        if isinstance(node, ast.Subscript):
            variable = self.variables.get(node.value.id if isinstance(node.value, ast.Name) else None)
            if variable is None or variable.kind not in ("input", "memory"):
                raise SyntaxError(f"cannot index {ast.unparse(node.value)}")
            return variable.lower, variable.upper
        if isinstance(node, ast.UnaryOp):
            low, high = self._range(node.operand)
            if isinstance(node.op, ast.USub):
                return -high, -low
            if isinstance(node.op, ast.Not):
                return 0, 1
            return low, high
        if isinstance(node, ast.BinOp):
            a, b = self._range(node.left), self._range(node.right)
            if isinstance(node.op, ast.Add):
                return a[0] + b[0], a[1] + b[1]
            if isinstance(node.op, ast.Sub):
                return a[0] - b[1], a[1] - b[0]
            if isinstance(node.op, ast.Mult):
                if (
                    self._fold(node.left) is None
                    and self._fold(node.right) is None
                    and (a[1] - a[0] + 1) * (b[1] - b[0] + 1) > TABLE_LIMIT
                ):
                    raise SyntaxError(f"{ast.unparse(node)}: operand ranges too wide for a table")
                products = [x * y for x in a for y in b]
                return min(products), max(products)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                values = [_apply(node.op, x, y) for x in range(a[0], a[1] + 1) for y in range(b[0], b[1] + 1) if y != 0]
                if not values or (a[1] - a[0] + 1) * (b[1] - b[0] + 1) > TABLE_LIMIT:
                    raise SyntaxError(f"{ast.unparse(node)}: operand ranges too wide for a table")
                return min(values), max(values)
            raise SyntaxError(f"unsupported operator in {ast.unparse(node)}")
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return 0, 1
        if isinstance(node, ast.IfExp):
            return _union(self._range(node.body), self._range(node.orelse))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            ranges = [self._range(a) for a in node.args]
            if name == "min":
                return min(r[0] for r in ranges), min(r[1] for r in ranges)
            if name == "max":
                return max(r[0] for r in ranges), max(r[1] for r in ranges)
            if name == "abs":
                low, high = ranges[0]
                return (0 if low <= 0 <= high else min(abs(low), abs(high))), max(abs(low), abs(high))
            if self.globals.get(name) is clip:
                return self._constant(node.args[1]), self._constant(node.args[2])
        raise SyntaxError(f"unsupported expression: {ast.unparse(node)}")

    def _builder(self, node, name):
        """A callback building `node`, clipped to the range of variable `name`.

        The clip is the identity on in-range values; it also tells the compiler's
        interval analysis the range, which can be looser than the one here.
        """
        build = self._expression(node)
        if name in self.variables:
            low, high = self.variables[name].lower, self.variables[name].upper
        else:  # _result and _out: known once lowering ends.
            return lambda state: build(state).clip(*self._special_range(name))
        return lambda state: build(state).clip(low, high)

    def _special_range(self, name):
        return (self.result_range if name == "_result" else self.output_range) or (0, 0)

    def _expression(self, node, truth=False):
        """A callback: application state -> IR expression for `node`."""
        lowering = self

        def build(state):
            value = lowering._build(node, state)
            if truth:
                low, high = lowering._range(node)
                if not (low >= 0 and high <= 1):
                    value = 1 - value.equals(0)
            return value

        return build

    def _build(self, node, state):
        folded = self._fold(node)
        if folded is not None:
            return constant(folded)
        if isinstance(node, ast.Name):
            # A field read back from the previous record is a token; the clip states its range.
            variable = self.variables[node.id]
            return state[node.id].clip(variable.lower, variable.upper)
        if isinstance(node, ast.Subscript):
            variable = self.variables[node.value.id]
            index = self._build(node.slice, state).clip(0, variable.length - 1)
            return read_at(index + (self.machine_payload + variable.offset)).clip(variable.lower, variable.upper)
        if isinstance(node, ast.UnaryOp):
            value = self._build(node.operand, state)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.Not):
                return value.equals(0)
            return value
        if isinstance(node, ast.BinOp):
            return self._binary(node, state)
        if isinstance(node, ast.Compare):
            result, left = None, node.left
            for op, right in zip(node.ops, node.comparators):
                term = _compare(op, self._build(left, state), self._build(right, state))
                result = term if result is None else select(result, term, 0)
                left = right
            return result
        if isinstance(node, ast.BoolOp):
            values = [self._truth(v, state) for v in node.values]
            result = values[0]
            for value in values[1:]:
                result = select(result, value, 0) if isinstance(node.op, ast.And) else select(result, 1, value)
            return result
        if isinstance(node, ast.IfExp):
            return select(self._truth(node.test, state), self._build(node.body, state), self._build(node.orelse, state))
        if isinstance(node, ast.Call):
            name, args = node.func.id, [self._build(a, state) for a in node.args]
            if name in ("min", "max"):
                result = args[0]
                for value in args[1:]:
                    smaller = (value - result).clip(0, 1) if name == "max" else (result - value).clip(0, 1)
                    result = select(smaller, value, result)
                return result
            if name == "abs":
                return args[0].relu() + (-args[0]).relu()
            return args[0].clip(self._constant(node.args[1]), self._constant(node.args[2]))
        raise SyntaxError(f"unsupported expression: {ast.unparse(node)}")

    def _truth(self, node, state):
        value = self._build(node, state)
        low, high = self._range(node)
        return value if low >= 0 and high <= 1 else 1 - value.equals(0)

    def _binary(self, node, state):
        left, right = self._build(node.left, state), self._build(node.right, state)
        a, b = self._range(node.left), self._range(node.right)
        fa, fb = self._fold(node.left), self._fold(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult) and (fa is not None or fb is not None):
            return right * fa if fa is not None else left * fb
        if isinstance(node.op, ast.Mod) and fb is not None and fb > 0 and a[0] >= 0:
            return left % fb
        # Other operations of two bounded variables become one lookup table.
        if (a[1] - a[0] + 1) * (b[1] - b[0] + 1) > TABLE_LIMIT:
            raise SyntaxError(f"{ast.unparse(node)}: operand ranges too wide for a table")
        span = b[1] - b[0] + 1
        table = tuple(
            _apply(node.op, x, y) if not (isinstance(node.op, (ast.FloorDiv, ast.Mod)) and y == 0) else 0
            for x in range(a[0], a[1] + 1)
            for y in range(b[0], b[1] + 1)
        )
        index = (left.clip(a[0], a[1]) - a[0]) * span + (right.clip(b[0], b[1]) - b[0])
        return index.lookup(table)

    # Assembling the machine.
    def _assemble(self):
        self.ops = _optimize(self.ops, self.text)
        memory = IndexedMemory(max(1, self.memory_cells), self.memory_limit, self.max_context, prefix="_memory_")
        state = Record("Program", tuple(self._fields()))
        self.machine = SequenceMachine(state, memory, self.ops, payload_capacity=self.payload)
        self.machine_payload = self.machine.payload_address

    def _fields(self):
        fields = []
        for name in self.fields:
            variable = self.variables[name]
            fields.append(Field(name, variable.lower, variable.upper))
        fields.append(Field("_out", *(self.output_range or (0, 0))))
        fields.append(Field("_result", *(self.result_range or (0, 0))))
        inputs = [v for v in self.variables.values() if v.kind == "input"]
        if inputs:  # Keeps prompt data inside the token alphabet.
            fields.append(Field("_data", min(v.lower for v in inputs), max(v.upper for v in inputs)))
        return fields

    def prompt(self, arguments):
        payload, state = [], {}
        for name in self.parameters:
            variable, value = self.variables[name], arguments[name]
            if variable.kind == "input":
                values = [int(v) for v in value]
                if len(values) != variable.length or any(not variable.lower <= v <= variable.upper for v in values):
                    raise ValueError(f"{name}: expected {variable.length} values in {variable.lower}..{variable.upper}")
                payload += values
            else:
                if not variable.lower <= int(value) <= variable.upper:
                    raise ValueError(f"{name}={value} outside {variable.lower}..{variable.upper}")
                state[name] = int(value)
        for f in self.machine.state.fields:
            state.setdefault(f.name, max(f.lower, min(0, f.upper)) if f.lower <= 0 <= f.upper else f.lower)
        return self.machine.initialize(payload, **state)

    def decode(self, trace):
        protocol = self.machine.protocol
        records = protocol.records(trace, include_initial=True, allow_partial=True)
        executable = self.machine.executable
        emits = {i for i, op in enumerate(executable) if id(op) in self.emits}
        outputs = [
            state["_out"]
            for (_, previous), (_, state) in itertools.pairwise(records)
            if previous["sequence_pc"] in emits
        ]
        halted = len(trace) > protocol.prefix_length and trace[-1] == protocol.halt_token
        final = records[-1][1]
        returned = (
            halted and isinstance(executable[final["sequence_pc"]], Halt) and self._returns_value(final["sequence_pc"])
        )
        return Run(
            final["_result"] if returned else None,
            outputs,
            len(records) - 1,
            len(trace) - protocol.prefix_length,
            halted,
        )

    def _returns_value(self, pc):
        executable = self.machine.executable
        return pc > 0 and isinstance(executable[pc - 1], Assign) and "_result" in executable[pc - 1].updates


def _load(target):
    node = ast.parse(ast.unparse(target), mode="eval").body
    return node


def _union(a, b):
    if a is None:
        return b
    return min(a[0], b[0]), max(a[1], b[1])


def _apply(op, x, y):
    if isinstance(op, ast.Mult):
        return x * y
    if isinstance(op, ast.FloorDiv):
        return x // y
    if isinstance(op, ast.Mod):
        return x % y
    if isinstance(op, ast.Add):
        return x + y
    if isinstance(op, ast.Sub):
        return x - y
    raise SyntaxError("unsupported operator")


def _compare(op, a, b):
    if isinstance(op, ast.Eq):
        return (a - b).equals(0)
    if isinstance(op, ast.NotEq):
        return 1 - (a - b).equals(0)
    if isinstance(op, ast.Lt):
        return (b - a).clip(0, 1)
    if isinstance(op, ast.LtE):
        return (b - a + 1).clip(0, 1)
    if isinstance(op, ast.Gt):
        return (a - b).clip(0, 1)
    if isinstance(op, ast.GtE):
        return (a - b + 1).clip(0, 1)
    raise SyntaxError("unsupported comparison")


def _final(name, labels, executable, seen=()):
    """Follow a label through any jumps it lands on."""
    index = labels[name]
    op = executable[index] if index < len(executable) else None
    if isinstance(op, Jump) and op.target not in seen:
        return _final(op.target, labels, executable, (*seen, name))
    return name


def _optimize(ops, text):
    """Thread jumps, drop jumps to the next operation, and drop unreachable operations."""
    while True:
        labels, position = {}, 0
        for op in ops:
            if isinstance(op, Label):
                labels[op.name] = position
            else:
                position += 1
        executable = [op for op in ops if not isinstance(op, Label)]

        changed, result, position, reachable = False, [], 0, True
        for _i, op in enumerate(ops):
            if isinstance(op, Label):
                result.append(op)
                reachable = True
                continue
            if not reachable:
                changed = True
                continue
            if isinstance(op, Jump):
                target = _final(op.target, labels, executable)
                if labels[target] == position + 1:
                    changed = True  # A jump to the next operation.
                    position += 1
                    reachable = True
                    continue
                if target != op.target:
                    op, changed = Jump(target), True
                reachable = False
            elif isinstance(op, Branch):
                when_true, when_false = (
                    _final(op.when_true, labels, executable),
                    _final(op.when_false, labels, executable),
                )
                if (when_true, when_false) != (op.when_true, op.when_false):
                    branch = Branch(op.condition, when_true, when_false)
                    text[id(branch)] = text[id(op)]
                    op, changed = branch, True
            elif isinstance(op, Halt):
                reachable = False
            result.append(op)
            position += 1
        if not changed:
            return _drop_unused_labels(result)
        ops = result


def _drop_unused_labels(ops):
    used = {op.target for op in ops if isinstance(op, Jump)}
    used |= {t for op in ops if isinstance(op, Branch) for t in (op.when_true, op.when_false)}
    return [op for op in ops if not isinstance(op, Label) or op.name in used]
