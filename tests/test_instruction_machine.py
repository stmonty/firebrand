import itertools

import pytest

from firebrand import Transformer, compile_machine
from firebrand.machines.instruction import (
    ADD,
    COUNTDOWN,
    HALT,
    JMP,
    JZ,
    LOAD,
    OUT,
    SET,
    STORE,
    SUB,
    InstructionSpec,
    reference_run,
)


@pytest.fixture(scope="module")
def compiled():
    spec = InstructionSpec()
    return spec, compile_machine(spec.machine())


def check(compiled, program, steps=70):
    spec, model = compiled
    trace = model.generate(spec.encode(program), spec.protocol.token_budget(steps))
    expected, outputs, halted = reference_run(program, value_limit=spec.value_limit, memory_cells=spec.memory_cells)
    assert halted and trace[-1] == -1
    assert spec.outputs(trace) == outputs
    records = spec.records(trace)
    by_address = dict(records)
    observed, emitted = [], []
    for base, r in records:
        if r["output_valid"]:
            emitted.append(r["output"])
        # Decode the write log independently, checking historical links.
        memory, seen, cursor = [0] * spec.memory_cells, set(), r["head"]
        while cursor:
            assert cursor <= base
            frame = by_address[cursor]
            key = frame["write_address"]
            if key not in seen:
                memory[key] = frame["write_value"]
                seen.add(key)
            assert frame["parent"] < cursor
            cursor = frame["parent"]
        if not r["searching"]:  # An instruction has finished, including LOAD.
            observed.append((r["pc"], r["acc"], memory, emitted.copy()))
    assert observed == expected
    return trace


def test_loop_and_overwriting(compiled):
    trace = check(compiled, COUNTDOWN)
    assert compiled[0].outputs(trace) == [3, 2, 1, 0]


def test_exhaustive_small_memory_aliases(compiled):
    for a, b, read in itertools.product(range(2), repeat=3):
        check(compiled, [(SET, 2), (STORE, a), (SET, 5), (STORE, b), (LOAD, read), (OUT, 0)])


def test_missing_key_walks_to_zero_and_newest_zero_wins(compiled):
    check(compiled, [(SET, 9), (STORE, 0), (LOAD, 1), (OUT, 0), (STORE, 0), (SET, 7), (LOAD, 0), (OUT, 0)])


def test_arithmetic_and_both_branch_outcomes(compiled):
    check(compiled, [(LOAD, 0), (JZ, 3), (SET, 9), (SET, 30), (ADD, 5), (OUT, 0), (JZ, 9), (SUB, 31), (OUT, 0)])
    check(compiled, [])
    check(compiled, [(HALT, 0), (OUT, 0)])


def test_checkpoint_and_uncached_equivalence(compiled, tmp_path):
    spec, model = compiled
    prefix, budget = spec.encode(COUNTDOWN), spec.protocol.token_budget(40)
    expected = model.generate(prefix, budget, use_cache=False)
    model.save(tmp_path)
    assert Transformer.load(tmp_path).generate(prefix, budget) == expected


def test_budget_stops_during_lookup_or_loop(compiled):
    spec, model = compiled
    program = [(SET, 7), (STORE, 0), (LOAD, 1), (OUT, 0)]
    trace = model.generate(spec.encode(program), spec.protocol.token_budget(3))
    assert trace[-1] != -1
    assert spec.records(trace)[-1][1]["searching"] == 1
    assert spec.outputs(trace) == []
    loop = model.generate(spec.encode([(JMP, 0)]), spec.protocol.token_budget(5))
    assert loop[-1] != -1 and len(spec.records(loop)) == 5


@pytest.mark.parametrize(
    "program",
    [
        [(9, 0)],
        [(SET, 32)],
        [(STORE, 4)],
        [(LOAD, -1)],
        [(JMP, 2)],
        [(OUT, 1)],
        [(HALT, 1)],
        [(SET, True)],
        [(HALT, 0)] * 16,
    ],
)
def test_invalid_programs(program):
    with pytest.raises((ValueError, TypeError)):
        InstructionSpec().encode(program)
