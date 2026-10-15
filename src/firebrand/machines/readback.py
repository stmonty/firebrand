"""Host-side closure readback and printing. Neither performs beta reduction."""

from ..ir import integer
from .lambda_calculus import App, Closure, Lam, Var


class ReadbackLimitError(ValueError):
    """Displaying a closure exceeded the chosen traversal budget."""


class _Budget:
    def __init__(self, max_nodes, max_depth):
        if integer(max_nodes) < 1 or integer(max_depth) < 1 or max_depth > 256:
            raise ValueError("positive node budget and depth in 1..256 required")
        self.remaining = max_nodes
        self.max_depth = max_depth

    def visit(self, depth):
        self.remaining -= 1
        if self.remaining < 0 or depth > self.max_depth:
            raise ReadbackLimitError("syntax traversal budget exceeded")


def readback(closure: Closure, *, max_nodes=10000, max_depth=128):
    """Substitute captured closures to produce a closed de Bruijn syntax tree.

    Locally bound indices stay unchanged. Captured closures are reconstructed in
    their own environments, producing closed subtrees which cannot be captured
    when inserted under a lambda. Unresolved free variables are errors. max_nodes
    counts visited source nodes, including variable occurrences being replaced.
    """
    budget = _Budget(max_nodes, max_depth)
    active = set()

    def expand(value, depth):
        if not isinstance(value, Closure):
            raise TypeError("expected a Closure")
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic closure environment")
        active.add(identity)
        try:
            return walk(value.term, value.environment, 0, depth)
        finally:
            active.remove(identity)

    def walk(term, environment, binders, depth):
        budget.visit(depth)
        if isinstance(term, Var):
            if term.index < binders:
                return term
            offset = term.index - binders
            if offset >= len(environment):
                raise ValueError("unresolved free variable in closure")
            return expand(environment[offset], depth + 1)
        if isinstance(term, Lam):
            return Lam(walk(term.body, environment, binders + 1, depth + 1))
        if isinstance(term, App):
            return App(
                walk(term.function, environment, binders, depth + 1),
                walk(term.argument, environment, binders, depth + 1),
            )
        raise TypeError("expected Var, Lam, or App")

    return expand(closure, 0)


def format_term(term, *, allow_open=False, max_nodes=10000, max_depth=128):
    """Print readable binder names; open indices optionally appear as #n."""
    budget = _Budget(max_nodes, max_depth)

    def render(node, binders, depth):
        budget.visit(depth)
        if isinstance(node, Var):
            if node.index < len(binders):
                return binders[-1 - node.index]
            if allow_open:
                return f"#{node.index}"
            raise ValueError("cannot print an open term without its environment")
        if isinstance(node, Lam):
            name = f"x{len(binders)}"
            return f"λ{name}. {render(node.body, binders + (name,), depth + 1)}"
        if isinstance(node, App):
            return f"({render(node.function, binders, depth + 1)}) ({render(node.argument, binders, depth + 1)})"
        raise TypeError("expected Var, Lam, or App")

    return render(term, (), 0)
