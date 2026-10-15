# firebrand

Compile small programs into transformer weights. No training required.

Describe a computation with bounded integers, and firebrand builds the attention
heads and ReLU layers that execute it. Examples include GCD, Conway's Life,
and an RV32IM interpreter used to run [DOOM](https://github.com/stmonty/firebrand/blob/main/docs/doom.md).

## Install

Requires Python 3.11+.

```sh
pip install firebrand
```

The core package uses NumPy and runs on a CPU. The faster C decoder also needs
`cc` (GCC or Clang) and a writable cache directory on Linux or macOS.

## Try it

```python
from firebrand import Machine, compile_machine, history

countdown = Machine((history() - 1).relu(), lower=0, upper=31, halt_values=(0,))
model = compile_machine(countdown)

print(model.generate([5]))
# [5, 4, 3, 2, 1, 0]
```

`history()` reads the last token. This machine subtracts one, stops at zero,
and returns the prompt followed by the generated tokens.

You can also compile a restricted Python function with `@program`, using
bounded integer parameters such as `Int[0, 60]`.

## Learn more

- [First program](https://github.com/stmonty/firebrand/blob/main/docs/tutorial.md): compile and run GCD.
- [Usage](https://github.com/stmonty/firebrand/blob/main/docs/usage.md): integer bounds, saving models, backends, and development.
- [Examples](https://github.com/stmonty/firebrand/tree/main/examples): memory, records, rendering, and more.
- [DOOM](https://github.com/stmonty/firebrand/blob/main/docs/doom.md): build and run the RISC-V demo.

Optional extras: `pip install "firebrand[gpu]"` for PyTorch and CUDA tools,
or `pip install "firebrand[llm]"` for Hugging Face export and the language-model bridge.

This is an alpha research library. Exact execution depends on the declared
integer ranges and compiled context limit; arbitrary Python is not supported.

## License

[MIT](https://github.com/stmonty/firebrand/blob/main/LICENSE).
