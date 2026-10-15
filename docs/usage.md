# Using firebrand

Start with the [GCD tutorial](tutorial.md). The core package needs Python 3.11+
and NumPy; neither a GPU nor a C compiler is needed for the Python backend.

## Machines and tokens

A `Machine` describes the next integer token as an expression over previous
tokens. `history()` reads the last token; `history(1)` reads the token before it.
`read_at(address)` reads an earlier absolute position. Expressions support
arithmetic, `.relu()`, `.clip(lo, hi)`, and `.lookup(table)`. `select`, `latest`,
and `match` support branches and retrieval.

```python
from firebrand import Machine, Transformer, compile_machine, history

machine = Machine((history() - 1).relu(), lower=0, upper=31, halt_values=(0,))
model = compile_machine(machine, max_context=128)
trace = model.generate([5], max_steps=20)
assert trace == [5, 4, 3, 2, 1, 0]

model.save("countdown")
restored = Transformer.load("countdown")
assert restored.generate([5]) == trace
```

The prompt must contain enough tokens for all history reads, and every token
must be an integer in the machine's inclusive range. `generate` returns the
prompt and generated tokens together. `max_steps` counts new tokens, and
`halt_values` stop generation. Reaching the step budget does not imply a halt.

`max_context` bounds the prefix the model can evaluate. Choose it when compiling;
running past that limit raises an error. The compiler supports at most 4,096
distinct token values and a context limit of at most `2**25`, with additional
checks on coefficients and intermediate ranges to preserve exact arithmetic.

Saving writes `config.json` and `weights.npz` into the destination, replacing
files with those names. Loading uses the saved weights without the source
program. It restores a `Transformer`, not the decorated function's `.run()` API.

## Choosing an interface

| Interface | Use it for |
| --- | --- |
| `Machine` | A next-token expression, such as a recurrence |
| `RecordMachine` | State expressed as typed records and transitions |
| `SequenceMachine` | Explicit reads, writes, assignments, and branches |
| `@program` | Restricted Python functions with bounded integers |

For `@program`, `Int[lo, hi]` declares integers, `Array[n, lo, hi]` declares
read-only input arrays, and `Memory(n, lo, hi)` creates writable memory.
`emit(x)` appends an output; `return` halts. See the [tutorial](tutorial.md) for
source availability, clipping, and budget behavior.

## Execution and export

`model.generate(prompt, max_steps=100)` defaults to the NumPy backend.
Pass `backend="c"` for the C decoder. Decorated programs use C by default;
pass `backend="python"` to `.run()` when no C compiler is available.

The C decoder requires Linux or macOS, `cc`, and a writable cache directory.
It builds the bundled `fastrt.c` on first use and reuses the shared library.

```sh
pip install "firebrand[gpu]"
pip install "firebrand[llm]"
```

| API | Purpose | Extra |
| --- | --- | --- |
| `firebrand.torch_export.to_torch(model)` | Export as a PyTorch module | `gpu` |
| `firebrand.gpu.Verifier(model).check(trace, start)` | Check a proposed trace in parallel on CUDA | `gpu` |
| `firebrand.gpu_generate.GPUGenerator(model)` | Generate tokens serially on CUDA | `gpu` |
| `firebrand.speculative.generate(model, prompt, draft, steps)` | Generate with a draft and CUDA verification | `gpu` |
| `firebrand.hf_export.to_huggingface(model, path)` | Write a Phi-3 checkpoint and tokenizer | `llm` |
| `firebrand.bridge.ExactArithmeticLM` | Add exact arithmetic to a language model | `llm` |

Installing the GPU extra does not provide an NVIDIA driver or guarantee CUDA
availability. PyTorch export can run on CPU; the CUDA tools require compatible
hardware, PyTorch, and Triton.

Hugging Face export is intended for small models. Checkpoints use dense float64
weights; keep that precision and use the supported attention configuration.
Prompts start with `<bos>`. The exporter checks numeric bounds because Phi-3's
RMSNorm uses float32 internally. See [the export tests](../tests/test_hf_export.py)
for complete loading and generation examples.

## Development

From a source checkout:

```sh
pip install -e ".[dev]"
pytest -q -rs
ruff check .
ruff format --check .
mypy
python -m build
python -m twine check --strict dist/*
```

Use `.[dev,gpu,llm]` to include optional integrations. Tests for optional
libraries or CUDA skip when their prerequisites are unavailable. Some RISC-V
tests require built binaries; see [DOOM](doom.md) for the toolchain setup.
On NixOS, run development commands inside `nix-shell`.
