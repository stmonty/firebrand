"""Post-execution trace inspection; these functions do not drive inference."""

from dataclasses import dataclass

from .lambda_calculus import App, Closure, Lam, State
from .readback import format_term, readback


@dataclass(frozen=True)
class RecordTrace:
    halted: bool
    records: tuple  # (address, field mapping), including initial state.
    partial_tokens: tuple


def inspect_records(definition, trace):
    """Separate complete records, observed halt, and a truncated final record."""
    records = tuple(definition.records(trace, include_initial=True, allow_partial=True))
    halted = len(trace) > definition.prefix_length and trace[-1] == definition.halt_token
    end = len(trace) - int(halted)
    count = (end - definition.prefix_length) % definition.record.width
    partial = tuple(trace[end - count : end]) if count else ()
    return RecordTrace(halted, records, partial)


@dataclass(frozen=True)
class LambdaEvent:
    step: int
    action: str
    record_address: int
    semantic_reads: tuple[int, ...]
    allocation: str | None
    before: State
    after: State
    changes: dict


@dataclass(frozen=True)
class LambdaTrace:
    records: RecordTrace
    events: tuple[LambdaEvent, ...]
    final_state: State

    def result_term(self, **limits):
        """Read back a confirmed halted result; never label a stopped run a result."""
        if not self.records.halted:
            raise ValueError("no result: execution stopped without a halt token")
        return readback(Closure(self.final_state.term, self.final_state.environment), **limits)


def inspect_lambda(spec, trace, terms):
    """Label observed transitions and their logical reads/allocations.

    Read addresses describe the abstract-machine operation, not every attention
    query: eager circuits also execute reads for branches that are not selected.
    This viewer does not replace differential verification of transition rules.
    """
    record_trace = inspect_records(spec.protocol, trace)
    states = spec.states(trace, terms)
    if record_trace.halted and not states[-1].halted:
        raise ValueError("halt token does not correspond to a weak-head state")
    events = []
    for index, ((_, before_record), (address, after_record)) in enumerate(
        zip(record_trace.records, record_trace.records[1:])
    ):
        before, after = states[index : index + 2]
        if before.halted:
            raise ValueError("trace continues after a weak-head state")
        allocation = None
        if before.searching:
            action = "LOOKUP_RESOLVE" if before.index == 0 else "LOOKUP_NEXT"
            reads = (before_record["environment"],)
        elif isinstance(before.term, App):
            action, allocation = "APP", "argument frame"
            reads = (before_record["term"],)
        elif isinstance(before.term, Lam):
            action, allocation = "BETA", "environment binding"
            reads = (before_record["term"], before_record["stack"])
        else:
            action = "LOOKUP_START"
            reads = (before_record["term"],)
        changes = {
            name: (before_record[name], value) for name, value in after_record.items() if value != before_record[name]
        }
        events.append(LambdaEvent(index + 1, action, address, reads, allocation, before, after, changes))
    return LambdaTrace(record_trace, tuple(events), states[-1])


def format_lambda_trace(report):
    """Compact semantic trace with addresses and before/after state changes."""
    lines = ["step  action          record  logical reads  allocation"]
    for event in report.events:
        allocation = f"{event.allocation} @{event.record_address}" if event.allocation else "—"
        lines.append(
            f"{event.step:4}  {event.action:14}  {event.record_address:6}  {event.semantic_reads!s:13}  {allocation}"
        )
        changes = ", ".join(f"{name}: {old}→{new}" for name, (old, new) in event.changes.items())
        lines.append(f"      {changes}")
    if report.records.halted:
        lines.append(f"HALT after {len(report.events)} completed transitions")
    else:
        lines.append(
            f"STOPPED WITHOUT HALT after {len(report.events)} completed transitions; "
            f"{len(report.records.partial_tokens)} partial-record tokens"
        )
    lines.append("Current term (open indices shown as #n): " + format_term(report.final_state.term, allow_open=True))
    lines.append(
        f"Captured bindings: {len(report.final_state.environment)}; pending arguments: {len(report.final_state.stack)}"
    )
    return "\n".join(lines)
