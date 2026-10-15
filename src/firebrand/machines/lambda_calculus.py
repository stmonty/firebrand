"""A bounded, call-by-name Krivine evaluator compiled using the generic IR.

The host serializes closed de Bruijn terms. During inference, generated records
carry control state, argument closures, and environment bindings. Evaluation stops
at weak-head form (a lambda with no pending arguments), not full normalization.
"""

from dataclasses import dataclass
from functools import lru_cache

from ..ir import integer, select
from ..machine import RecordMachine, Transition
from ..records import Field, Record

VAR, LAM, APP = 1, 2, 3


@dataclass(frozen=True)
class Var:
    index: int

    def __post_init__(self):
        if integer(self.index) < 0:
            raise ValueError("de Bruijn index must be nonnegative")


@dataclass(frozen=True)
class Lam:
    body: object


@dataclass(frozen=True)
class App:
    function: object
    argument: object


def validate_closed(term, depth=0):
    if isinstance(term, Var):
        if term.index >= depth:
            raise ValueError("term has a free variable")
    elif isinstance(term, Lam):
        validate_closed(term.body, depth + 1)
    elif isinstance(term, App):
        validate_closed(term.function, depth)
        validate_closed(term.argument, depth)
    else:
        raise TypeError("expected Var, Lam, or App")


@dataclass(frozen=True)
class Closure:
    term: object
    environment: tuple = ()


@dataclass(frozen=True)
class State:
    term: object
    environment: tuple = ()
    stack: tuple = ()
    searching: bool = False
    index: int = 0

    @property
    def halted(self):
        return not self.searching and isinstance(self.term, Lam) and not self.stack


def reference_states(term, max_steps=24):
    """Independent object/tuple evaluator; no IR, matrices, or address layout."""
    validate_closed(term)
    if integer(max_steps) < 0:
        raise ValueError("negative step budget")
    state = State(term)
    states = [state]
    for _ in range(max_steps):
        if state.halted:
            break
        term, env, stack = state.term, state.environment, state.stack
        if state.searching:
            if not env:
                raise ValueError("unbound variable during lookup")
            if state.index:
                state = State(term, env[1:], stack, True, state.index - 1)
            else:
                closure = env[0]
                state = State(closure.term, closure.environment, stack)
        elif isinstance(term, App):
            state = State(term.function, env, (Closure(term.argument, env),) + stack)
        elif isinstance(term, Lam):
            state = State(term.body, (stack[0],) + env, stack[1:])
        else:
            state = State(term, env, stack, True, term.index)
        states.append(state)
    return states


@dataclass(frozen=True)
class LambdaSpec:
    node_capacity: int = 16
    max_context: int = 256

    def __post_init__(self):
        integer(self.node_capacity)
        integer(self.max_context)
        if self.node_capacity < 8 or self.node_capacity % 8:
            raise ValueError("node_capacity must be a positive multiple of eight")
        if self.max_context < self.prefix_length:
            raise ValueError("context must fit the input prefix")

    @property
    def prefix_length(self):
        return 16 + 3 * self.node_capacity

    @property
    def node(self):
        return Record(
            "Term", (Field("tag", VAR, APP), Field("first", 0, self.max_context), Field("second", 0, self.max_context))
        )

    @property
    def frame(self):
        names = ("term", "environment", "stack", "searching", "index", "closure_term", "closure_environment", "parent")
        return Record("LambdaFrame", tuple(Field(name, 0, self.max_context) for name in names))

    @property
    def protocol(self):
        return RecordMachine(self.frame, self.prefix_length, self.max_context, -1, self.max_context)

    def serialize(self, term):
        """Syntax only: return prompt tokens and an address-to-AST map."""
        validate_closed(term)
        tokens, addresses, terms = [0] * 8, {}, {}

        def visit(node):
            if node in addresses:
                return addresses[node]
            if isinstance(node, Var):
                record = self.node.pack(tag=VAR, first=node.index, second=0)
            elif isinstance(node, Lam):
                record = self.node.pack(tag=LAM, first=visit(node.body), second=0)
            else:
                record = self.node.pack(tag=APP, first=visit(node.function), second=visit(node.argument))
            if len(addresses) >= self.node_capacity:
                raise ValueError("term exceeds node capacity")
            address = len(tokens)
            tokens.extend(record)
            addresses[node], terms[address] = address, node
            return address

        root = visit(term)
        tokens = self.protocol.initialize(
            tokens,
            term=root,
            environment=0,
            stack=0,
            searching=0,
            index=0,
            closure_term=0,
            closure_environment=0,
            parent=0,
        )
        return tokens, terms

    def machine(self):
        return self.protocol.lower_machine(self.transition)

    def transition(self, step):
        state = step.state
        base = step.address
        term, env, stack = state["term"], state["environment"], state["stack"]
        tag = self.node.read(term, "tag")
        first, second = self.node.read(term, "first"), self.node.read(term, "second")
        argument = {name: self.frame.read(stack, name) for name in ("closure_term", "closure_environment", "parent")}
        binding = {name: self.frame.read(env, name) for name in ("closure_term", "closure_environment", "parent")}
        evaluating = state["searching"].equals(0)
        is_app, is_lam, is_var = tag.equals(APP), tag.equals(LAM), tag.equals(VAR)
        resolves = state["index"].equals(0)
        halted = select(evaluating, select(is_lam, stack.equals(0), 0), 0)

        eval_term = select(is_var, term, first)
        lookup_term = select(resolves, binding["closure_term"], term)
        next_term = select(evaluating, eval_term, lookup_term)
        next_env = select(
            evaluating, select(is_lam, base, env), select(resolves, binding["closure_environment"], binding["parent"])
        )
        next_stack = select(evaluating, select(is_app, base, select(is_lam, argument["parent"], stack)), stack)
        next_search = select(evaluating, is_var, select(resolves, 0, 1))
        next_index = select(evaluating, select(is_var, first, 0), select(resolves, 0, (state["index"] - 1).relu()))
        # APP allocates an argument frame; BETA allocates an environment frame.
        # All other transitions publish zeros in these unused allocation fields.
        alloc_term = select(evaluating, select(is_app, second, select(is_lam, argument["closure_term"], 0)), 0)
        alloc_env = select(evaluating, select(is_app, env, select(is_lam, argument["closure_environment"], 0)), 0)
        alloc_parent = select(evaluating, select(is_app, stack, select(is_lam, env, 0)), 0)
        return Transition(
            dict(
                term=next_term,
                environment=next_env,
                stack=next_stack,
                searching=next_search,
                index=next_index,
                closure_term=alloc_term,
                closure_environment=alloc_env,
                parent=alloc_parent,
            ),
            halt=halted,
        )

    def token_budget(self, max_steps):
        return self.protocol.token_budget(max_steps)

    def states(self, trace, terms):
        """Decode completed states for inspection, including the initial state.

        A budget can stop after the first token of an unfinished next transition;
        that partial record is ignored. Completion is indicated only by -1.
        """

        records = self.protocol.records(trace, include_initial=True, allow_partial=True)
        frames = dict(records)

        def frame(address):
            if address < self.prefix_length or address not in frames:
                raise ValueError("invalid or incomplete closure-frame address")
            record = frames[address]
            for name in ("parent", "closure_environment"):
                pointer = record[name]
                if pointer and (pointer >= address or pointer < self.prefix_length or pointer not in frames):
                    raise ValueError("closure-frame links must point to earlier complete records")
            if record["closure_term"] not in terms:
                raise ValueError("unknown closure term address")
            return record

        @lru_cache(None)
        def environment(address):
            if not address:
                return ()
            record = frame(address)
            return (Closure(terms[record["closure_term"]], environment(record["closure_environment"])),) + environment(
                record["parent"]
            )

        def arguments(address):
            result = []
            while address:
                record = frame(address)
                result.append(Closure(terms[record["closure_term"]], environment(record["closure_environment"])))
                address = record["parent"]
            return tuple(result)

        result = []
        for address, record in records:
            if record["term"] not in terms or record["searching"] not in (0, 1):
                raise ValueError("invalid term address or lookup mode")
            if record["environment"] > address or record["stack"] > address:
                raise ValueError("state points to a future closure frame")
            result.append(
                State(
                    terms[record["term"]],
                    environment(record["environment"]),
                    arguments(record["stack"]),
                    bool(record["searching"]),
                    record["index"],
                )
            )
        return result
