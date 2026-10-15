# Your first compiled program

firebrand turns a restricted Python function into transformer weights. This
example computes the greatest common divisor by repeated subtraction.

Install the package with `pip install firebrand`, then save this as `gcd_demo.py`:

```python
from firebrand import Int, program


@program
def gcd(a: Int[0, 60], b: Int[0, 60]):
    while a != 0 and b != 0:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a + b


# An ordinary call runs the original Python function.
print(gcd(60, 48))

# .run() executes the compiled transformer. Use the NumPy backend here
# so that this example does not require a C compiler.
run = gcd.run(60, 48, backend="python")
assert run.halted
assert run.result == gcd(60, 48)
print(run.result)
```

Run it with `python gcd_demo.py`. It prints `12` twice. Keep the function in a
file: the decorator reads its source, so a definition entered into a plain
Python REPL or `python -c` will not work.

## What gets compiled?

`Int[0, 60]` declares an inclusive range. The compiler uses these bounds to
construct the model. Arguments must fit the declared ranges.

The loop, branches, and assignments become operations of a `SequenceMachine`.
Those operations are represented as records of integer tokens. The transformer
generates the records until it reaches the return instruction or its budget.
The model is built lazily on first access to `gcd.model` or first `.run()`.

Add these lines to inspect the operations and the execution cost:

```python
print(gcd.listing())
print(run.steps, run.tokens, run.halted)
```

`run.result` is the returned value, `run.outputs` contains values passed to
`emit`, and `run.halted` tells you whether the program finished. Always check
`halted` before treating a run as a completed computation.

## Run with the C decoder

If GCC or Clang is available as `cc`, replace the call with:

```python
run = gcd.run(60, 48, backend="c")
assert run.halted and run.result == 12
```

This is the default backend for `.run()`. On first use it compiles the bundled
C source into `~/.cache/firebrand` (or `$XDG_CACHE_HOME/firebrand`).

## Bounds and budgets

Start with small ranges. Larger ranges and lookup tables can make the compiled
model much larger. This is a restricted language: arbitrary imports, library
calls, and Python objects inside the function are not supported.

Assignments in the compiled program are clipped to the variable's declared
range. Ordinary Python calls do not apply this clipping, so the two executions
can differ if your program exceeds its bounds. This GCD example stays in range.

`@program(max_context=16384)` sets the context budget (also the default).
`gcd.run(..., max_steps=100)` limits execution to at most 100 machine steps;
a step can use several tokens. A budget that is too small can return a run with
`halted=False`. Increasing the budget does not change the compiled integer ranges.

## Next steps

Try other inputs between 0 and 60 and compare `.run(...).result` with an
ordinary call. Then explore [usage and backends](usage.md) or the repository's
[examples](../examples), including read-only arrays, writable memory, and records.
