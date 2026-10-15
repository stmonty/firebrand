"""Restricted Python functions compiled with @program behave as Python does."""

import random

import pytest

from firebrand.program import Array, Int, Memory, emit, program


@program
def gcd(a: Int[0, 60], b: Int[0, 60]):
    while a != 0 and b != 0:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a + b


@program(max_context=1 << 15)
def fibonacci(n: Int[0, 15]):
    a: Int[0, 1000] = 0
    b: Int[0, 1000] = 1
    for _i in range(n):
        emit(a)
        a, b = b, a + b


@program
def histogram(data: Array[16, 0, 4]):
    counts = Memory(5, 0, 16)
    for i in range(16):
        counts[data[i]] += 1
    for k in range(5):
        emit(counts[k])


@program(max_context=1 << 16)
def sieve(n: Int[2, 60]):
    composite = Memory(61, 0, 1)
    for p in range(2, n + 1):
        if composite[p] == 0:
            emit(p)
            multiple: Int[0, 120] = p * 2
            while multiple <= n:
                composite[multiple] = 1
                multiple += p


@program(max_context=1 << 16)
def bubble_sort(values: Array[8, 0, 99]):
    items = Memory(8, 0, 99)
    for i in range(8):
        items[i] = values[i]
    for limit in range(7, 0, -1):
        for j in range(limit):
            left: Int[0, 99] = items[j]
            right: Int[0, 99] = items[j + 1]
            if left > right:
                items[j] = right
                items[j + 1] = left
    for i in range(8):
        emit(items[i])


@program(max_context=1 << 15)
def collatz(n: Int[1, 30]):
    steps: Int[0, 200] = 0
    x: Int[1, 3000] = n
    while x != 1:
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        steps += 1
    return steps


@program
def extremes(values: Array[10, -50, 50]):
    low: Int[-50, 50] = 50
    high: Int[-50, 50] = -50
    largest: Int[0, 50] = 0
    for i in range(10):
        low = min(low, values[i])
        high = max(high, values[i])
        largest = max(largest, abs(values[i]))
    emit(low)
    emit(high)
    return largest


@program
def dot(a: Array[6, 0, 9], b: Array[6, 0, 9]):
    total: Int[0, 486] = 0
    for i in range(6):
        total += a[i] * b[i]
    return total


@program
def first_over(values: Array[12, 0, 20], threshold: Int[0, 20]):
    count: Int[0, 12] = 0
    found: Int[-1, 11] = -1
    for i in range(12):
        if values[i] == 0:
            continue
        count += 1
        if values[i] > threshold:
            found = i
            break
    emit(count)
    return found


def check(compiled, *args):
    expected_result, expected_outputs = compiled.reference(*args)
    run = compiled.run(*args)
    assert run.halted
    assert (run.result, run.outputs) == (expected_result, expected_outputs), (args, run)


def test_gcd_on_every_small_pair():
    for a in range(0, 61, 7):
        for b in range(0, 61, 5):
            check(gcd, a, b)


def test_loops_and_output():
    for n in (0, 1, 7, 15):
        check(fibonacci, n)
    for n in (2, 10, 37, 60):
        check(sieve, n)
    for n in range(1, 31):
        check(collatz, n)


@pytest.mark.parametrize("seed", range(5))
def test_arrays_and_memory(seed):
    rng = random.Random(seed)
    check(histogram, [rng.randint(0, 4) for _ in range(16)])
    check(bubble_sort, [rng.randint(0, 99) for _ in range(8)])
    check(extremes, [rng.randint(-50, 50) for _ in range(10)])
    check(dot, [rng.randint(0, 9) for _ in range(6)], [rng.randint(0, 9) for _ in range(6)])
    check(first_over, [rng.choice([0, 0, rng.randint(0, 20)]) for _ in range(12)], rng.randint(0, 20))


def test_python_and_the_transformer_agree_on_the_example():
    assert gcd(12, 8) == 4 == gcd.run(12, 8).result
    assert sieve.reference(20)[1] == [2, 3, 5, 7, 11, 13, 17, 19]


def test_listing_shows_one_step_per_line():
    text = gcd.listing()
    assert "if a > b" in text and "return a + b" in text
    assert len([line for line in text.splitlines() if not line.endswith(":")]) == len(gcd.machine.executable)


def test_assignments_that_may_leave_their_range_are_clipped():
    @program
    def saturate(a: Int[0, 9]):
        b: Int[0, 9] = a + 5
        return b

    assert [saturate.run(a).result for a in (0, 4, 9)] == [5, 9, 9]
    assert saturate(9) == 14  # Plain Python does not clip: the run reveals the overflow.


@pytest.mark.parametrize(
    "source, message",
    [
        ("def f(a):\n    return a\n", "needs a type"),
        ("def f(a: Int[0, 9]):\n    b = a\n", "needs a declaration"),
        ("def f(a: Int[0, 9]):\n    print(a)\n", "unsupported statement"),
        ("def f(a: Int[0, 999], b: Int[0, 999]):\n    c: Int[0, 9] = a * b\n", "too wide"),
    ],
)
def test_clear_errors(source, message, tmp_path):
    import importlib.util

    path = tmp_path / "bad.py"
    path.write_text("from firebrand.program import Int, program\n" + source)
    spec = importlib.util.spec_from_file_location("bad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SyntaxError, match=message):
        program(module.f)


def test_functions_without_source_are_refused():
    namespace = {"Int": Int, "program": program}
    exec("def f(a: Int[0, 3]):\n    return a\n", namespace)
    with pytest.raises(TypeError, match="cannot find"):
        program(namespace["f"])
