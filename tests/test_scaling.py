"""Positional phase features, single-head latest-write lookup, and layer packing."""

import itertools

import numpy as np
import pytest

from firebrand import (
    Assign,
    Branch,
    Field,
    Halt,
    IndexedMemory,
    Jump,
    Label,
    Machine,
    Read,
    Record,
    SequenceMachine,
    Transformer,
    Write,
    WriteLog,
    compile_machine,
    history,
    latest,
    position_mod,
    read_at,
    select,
)
from firebrand.ir import nodes
from firebrand.machines.gcd import gcd_machine
from firebrand.machines.instruction import COUNTDOWN, InstructionSpec
from firebrand.machines.lambda_calculus import App, Lam, LambdaSpec, Var
from firebrand.machines.pixel import DOWN, LEFT, RIGHT
from firebrand.machines.scheduled_histogram import ScheduledHistogramSpec
from firebrand.machines.stack import POP, PUSH, StackSpec
from firebrand.machines.tile import TileMachineSpec
from firebrand.reference import evaluate


def assert_slots_match_reference(model, expr, prefix):
    logits, probes = model.logits(list(prefix), probes=True)
    assert int(np.argmax(logits)) + model.config["lower"] == evaluate(expr, list(prefix))
    for node, entry in zip(nodes(expr), model.config["layout"]):
        assert probes[entry["stage"]][entry["slot"]] == evaluate(node, list(prefix))


@pytest.mark.parametrize("pack", [False, True])
def test_position_mod_matches_reference(pack):
    expr = (position_mod(3, 1) + position_mod(5) + history()).clip(0, 9)
    model = compile_machine(Machine(expr, 0, 9), max_context=64, pack=pack)
    for length in range(1, 17):
        assert_slots_match_reference(model, expr, [length % 10] * length)


def test_record_context_is_not_capped_by_record_width():
    # The old phase circuit needed max_context // width threshold pairs, which
    # failed just above 45,000 tokens for 11-field records.
    large = TileMachineSpec(max_context=2**20)
    small = TileMachineSpec()
    tiles = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    actions = [RIGHT, DOWN, LEFT]
    expected = compile_machine(small.machine()).generate(small.encode(actions, tiles), small.token_budget(len(actions)))
    model = compile_machine(large.machine())
    assert model.config["max_context"] == 2**20
    assert model.generate(large.encode(actions, tiles), large.token_budget(len(actions))) == expected


def lookup_expr():
    # Tagged positions are 1, 3, 5, ... from position 1; keys come from history.
    return read_at(latest(history(), period=2, phase=1, start=1))


@pytest.mark.parametrize("pack", [False, True])
def test_latest_matches_reference_exhaustively(pack):
    expr = lookup_expr()
    model = compile_machine(Machine(expr, -2, 2), max_context=16, pack=pack)
    for length in range(2, 6):
        for prefix in itertools.product(range(-2, 3), repeat=length):
            assert_slots_match_reference(model, expr, prefix)


def test_latest_semantics():
    at = lambda prefix: evaluate(latest(history(), period=2, phase=1, start=1), prefix)
    assert at([0, 2, 0, 2, 0, 1, 2]) == 3  # Newest exact match wins.
    assert at([0, 1, 0, -2, 2]) == 1  # No match: nearest value.
    assert at([0, 1, 0, 3, 2]) == 3  # Equal distance: newest.
    model = compile_machine(Machine(lookup_expr(), -2, 2), max_context=16)
    with pytest.raises(ValueError, match="tagged"):
        model.logits([2])
    with pytest.raises(ValueError, match="tagged"):
        evaluate(lookup_expr(), [2])


def test_latest_contracts(tmp_path):
    with pytest.raises(ValueError, match="tag"):
        latest(history(), period=2, phase=2, start=0)
    with pytest.raises(ValueError, match="tags"):
        compile_machine(Machine(lookup_expr(), 0, 2), max_context=1)
    with pytest.raises(ValueError, match="float64"):
        compile_machine(
            Machine(read_at(latest(history(), period=1, phase=0, start=0)), 60000, 64000), max_context=2**20
        )
    model = compile_machine(Machine(lookup_expr(), -2, 2), max_context=16)
    prefix = [0, 1, 0, 2, 1]
    expected = model.generate(prefix, 8, use_cache=False)
    assert model.generate(prefix, 8) == expected
    model.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix, 8) == expected


def memory_program(memory, capacity=3):
    """Replay payload operations (kind, address, value); kind 1 writes."""
    state = Record("Replay", (Field("i", 0, capacity), Field("n", 0, capacity), Field("last", 0, memory.value_limit)))
    payload = state.width + 1 + len(memory.fields)  # After the zero record.

    def operand(offset, upper):
        return lambda s: read_at(payload + 3 * s["i"].clip(0, capacity - 1) + offset).clip(0, upper)

    advance = Assign({"i": lambda s: (s["i"] + 1).clip(0, capacity)})
    return SequenceMachine(
        state,
        memory,
        (
            Label("top"),
            Branch(lambda s: (s["i"] - s["n"]).equals(0), "done", "dispatch"),
            Label("dispatch"),
            Branch(lambda s: operand(0, 1)(s), "write", "read"),
            Label("read"),
            Read("last", operand(1, memory.cells - 1)),
            advance,
            Jump("top"),
            Label("write"),
            Write(operand(1, memory.cells - 1), operand(2, memory.value_limit)),
            advance,
            Jump("top"),
            Label("done"),
            Halt(),
        ),
        payload_capacity=3 * capacity,
    )


@pytest.mark.parametrize("memory", [WriteLog(2, 1, 1024), IndexedMemory(2, 1, 1024)])
def test_indexed_memory_matches_dictionary_and_write_log(memory):
    program = memory_program(memory)
    model = compile_machine(program.lower_machine())
    read = program.labels["read"]
    operations = [(0, a, 0) for a in range(2)] + [(1, a, v) for a in range(2) for v in range(2)]
    # The linked write log is covered elsewhere; a shorter cross-check keeps it fast.
    for size in range(4 if isinstance(memory, IndexedMemory) else 3):
        for ops in itertools.product(operations, repeat=size):
            trace = model.generate(
                program.initialize([x for op in ops for x in op], i=0, n=size, last=0),
                program.protocol.token_budget(8 * size + 4),
            )
            assert trace[-1] == -1
            cells, expected = {}, []
            for kind, address, value in ops:
                if kind:
                    cells[address] = value
                else:
                    expected.append(cells.get(address, 0))
            records = program.protocol.records(trace, include_initial=True)
            observed = [
                state["last"]
                for (_, previous), (_, state) in itertools.pairwise(records)
                if previous["sequence_pc"] == read and state["sequence_pc"] == read + 1
            ]
            assert observed == expected


def test_indexed_memory_is_cheaper_and_checks_layout():
    costs = {}
    for indexed in (False, True):
        spec = ScheduledHistogramSpec(capacity=5, max_context=1024, indexed=indexed)
        model = compile_machine(spec.machine())
        trace = model.generate(spec.encode([1, 0, 1, 2, 1]), spec.protocol.token_budget(60))
        assert spec.result(trace) == [1, 3, 1]
        costs[indexed] = (
            len(spec.protocol.records(trace)),
            len(trace) - spec.protocol.prefix_length,
            model.config["upper"] - model.config["lower"] + 1,
        )
    assert costs == {False: (31, 342, 1026), True: (20, 121, 7)}
    memory = IndexedMemory(2, 2, 64)
    record = Record("Bad", memory.fields + (Field("x", 0, 1),))
    step = type("Step", (), {"prefix_length": record.width})()
    with pytest.raises(ValueError, match="final two"):
        memory.step(step, record)


def applications():
    stack, lam = StackSpec(), LambdaSpec()
    tile, instruction = TileMachineSpec(), InstructionSpec()
    gcd = gcd_machine()
    histogram = ScheduledHistogramSpec(capacity=3, max_context=512, indexed=True)
    identity = Lam(Var(0))
    return [
        ("stack", stack.machine(), stack.encode([(PUSH, 3), (PUSH, 5), (POP, 0)]), 13),
        ("lambda", lam.machine(), lam.serialize(App(identity, identity))[0], lam.token_budget(16)),
        ("tile", tile.machine(), tile.encode([RIGHT, DOWN], [[0, 1, 0, 0], [0] * 4, [0] * 4]), tile.token_budget(2)),
        ("instruction", instruction.machine(), instruction.encode(COUNTDOWN), instruction.protocol.token_budget(30)),
        ("gcd", gcd.lower_machine(), gcd.initialize(a=12, b=8, result=0), gcd.protocol.token_budget(40)),
        ("histogram", histogram.machine(), histogram.encode([1, 0, 1]), histogram.protocol.token_budget(20)),
    ]


@pytest.mark.parametrize("name,machine,prefix,budget", applications())
def test_packed_models_execute_identically(name, machine, prefix, budget, tmp_path):
    plain = compile_machine(machine, pack=False)
    packed = compile_machine(machine)
    expected = plain.generate(prefix, budget)
    assert packed.generate(prefix, budget) == expected
    assert packed.generate(prefix, budget, use_cache=False) == expected
    assert len(packed.config["schedule"]) < len(plain.config["schedule"])
    packed.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix, budget) == expected


@pytest.mark.parametrize(
    "expr",
    [
        (2 * history() + history(1) - 3).clip(-4, 4),
        select(history().equals(0), read_at(history(1).clip(0, 1)), history(1)),
        read_at(read_at(history().clip(0, 1)).clip(0, 1)),
    ],
)
def test_packing_preserves_every_slot(expr):
    machine = Machine(expr, -4, 4)
    model = compile_machine(machine)
    for prefix in itertools.product(range(-4, 5), repeat=max(2, machine.required_prefix)):
        try:
            evaluate(expr, list(prefix))
        except ValueError:
            continue
        assert_slots_match_reference(model, expr, prefix)


@pytest.mark.parametrize("name,machine,prefix,budget", applications()[:4])
def test_sparse_storage_executes_identically(name, machine, prefix, budget, tmp_path, monkeypatch):
    import firebrand.compiler as compiler
    from firebrand.runtime import Sparse

    dense = compile_machine(machine)
    monkeypatch.setattr(compiler, "DENSE_LIMIT", 0)
    sparse = compile_machine(machine)
    assert sparse.config["format_version"] == 4
    assert isinstance(sparse.weights["ff_0_in"], Sparse)
    assert sparse.weights["embedding"].shape[1] == sparse.config["input_width"]
    expected = dense.generate(prefix, budget)
    assert sparse.generate(prefix, budget) == expected
    assert sparse.generate(prefix, budget, use_cache=False) == expected
    np.testing.assert_array_equal(sparse.logits(prefix), dense.logits(prefix))
    sparse.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix, budget) == expected


def test_lookup_tables_are_exact():
    table = [5, 5, -3, 0, 7, 7, 7, 1]
    expr = (history().clip(-2, 5) + 2).lookup(table, lower=0)
    model = compile_machine(Machine(expr.clip(-4, 7), -4, 7), max_context=16)
    for x in range(-2, 6):
        assert model.generate([x], 1)[-1] == table[x + 2]
    with pytest.raises(ValueError, match="table"):
        compile_machine(Machine((history().clip(-2, 5) + 3).lookup(table).clip(-4, 7), -4, 7))


@pytest.mark.parametrize("name,machine,prefix,budget", applications())
def test_compiled_decoder_matches_numpy(name, machine, prefix, budget):
    for model in (compile_machine(machine), compile_machine(machine, pack=False)):
        assert model.generate(prefix, budget, backend="c") == model.generate(prefix, budget)


def test_compiled_decoder_errors_and_scanning_heads():
    pointer = compile_machine(Machine(read_at(history()), 0, 7, (0,)))
    with pytest.raises(ValueError, match="address"):
        pointer.generate([7], 1, backend="c")
    model = compile_machine(Machine(-history(), -3, 3), max_context=8)
    assert model.generate([3], 8, backend="c") == model.generate([3], 8)
    with pytest.raises(ValueError, match="context"):
        model.generate([3], 9, backend="c")
    with pytest.raises(ValueError, match="tagged"):
        compile_machine(Machine(lookup_expr(), -2, 2), max_context=16).generate([2], 1, backend="c")
    # Version-1 checkpoints have no addressing contract, so their heads scan.
    legacy = compile_machine(Machine(history(2), -4, 4), pack=False)
    legacy.config.pop("addressing")
    assert legacy.generate([1, 2, 3, -4], 4, backend="c") == [1, 2, 3, -4, 2, 3, -4, 2]
