"""Drafts for speculative decoding: a Machine's expression graph compiled to C.

Draft(machine).run(prefix, max_steps) proposes the tokens the compiled model
would generate, following reference.py's semantics, much faster than running the
transformer serially. If the source program fails (an address outside the
prefix, a select condition that is not 0 or 1, a token outside the alphabet),
run() stops there and sets `error`.
"""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from .ir import Expr, Machine, nodes

ERRORS = {
    1: "insufficient history",
    2: "address outside causal prefix",
    3: "select condition must be 0 or 1",
    4: "lookup argument outside its table",
    5: "mod requires a nonnegative operand",
    6: "no sentinel in causal prefix",
    7: "no tagged key in causal prefix",
    8: "transition outside alphabet",
    9: "out of memory",
}

_PRELUDE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t *keys; int64_t *positions; int64_t capacity, used; } Map;

static uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return k; }

static int map_put(Map *m, uint64_t key, int64_t position) {
    if (2 * (m->used + 1) > m->capacity) {
        int64_t capacity = m->capacity ? 2 * m->capacity : 1024;
        uint64_t *keys = malloc(sizeof(uint64_t) * capacity);
        int64_t *positions = malloc(sizeof(int64_t) * capacity);
        if (!keys || !positions) { free(keys); free(positions); return 1; }
        for (int64_t i = 0; i < capacity; i++) positions[i] = -1;
        for (int64_t i = 0; i < m->capacity; i++) {
            if (m->positions[i] < 0) continue;
            int64_t s = mix(m->keys[i]) & (capacity - 1);
            while (positions[s] >= 0) s = (s + 1) & (capacity - 1);
            keys[s] = m->keys[i]; positions[s] = m->positions[i];
        }
        free(m->keys); free(m->positions);
        m->keys = keys; m->positions = positions; m->capacity = capacity;
    }
    int64_t s = mix(key) & (m->capacity - 1);
    while (m->positions[s] >= 0 && m->keys[s] != key) s = (s + 1) & (m->capacity - 1);
    if (m->positions[s] < 0) m->used++;
    m->keys[s] = key; m->positions[s] = position;
    return 0;
}

static int64_t map_get(const Map *m, uint64_t key) {
    if (!m->capacity) return -1;
    int64_t s = mix(key) & (m->capacity - 1);
    while (m->positions[s] >= 0) {
        if (m->keys[s] == key) return m->positions[s];
        s = (s + 1) & (m->capacity - 1);
    }
    return -1;
}

static int64_t floor_mod(int64_t a, int64_t b) { int64_t r = a % b; return r < 0 ? r + b : r; }
#define FAIL(code) do { *error = code; return 0; } while (0)
"""


def _literal(value):
    return f"INT64_C({value})"


class Draft:
    """A Machine's next-token function compiled to native code."""

    def __init__(self, machine: Machine):
        self.machine = machine
        root = machine.next_value
        order = nodes(root)
        self.index = {node: i for i, node in enumerate(order)}
        self.matches = [n for n in order if n.op == "match"]
        self.latests = [n for n in order if n.op == "latest"]
        for n in self.matches:
            if n.args[0] > 4:
                raise ValueError("drafts support match keys of at most four digits")
        source = self._source(order, root)
        self._library = _build(source)
        self._run = self._library.draft_run
        self._run.restype = ctypes.c_int64
        self._run.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.error = None

    # -- Code generation -------------------------------------------------------

    def _value(self, arg):
        return f"v[{self.index[arg]}]" if isinstance(arg, Expr) else _literal(arg)

    def _node(self, node, tables):
        """C statements computing node into v[i], for the prefix tokens[0:L]."""
        i, op, args = self.index[node], node.op, node.args
        a = [self._value(x) for x in args] if op not in ("match", "lookup") else None
        out = f"v[{i}]"
        if op == "constant":
            return f"{out} = {_literal(args[0])};"
        if op == "history":
            return f"if (L <= {args[0]}) FAIL(1); {out} = tokens[L - 1 - {args[0]}];"
        if op == "position":
            return f"{out} = L - 1;"
        if op == "position_mod":
            return f"{out} = floor_mod(L - 1 + {args[1]}, {args[0]});"
        if op == "position_digit":
            return f"{{ int64_t s = L - 1 + {args[1]}; {out} = s >= 0 ? (s >> {8 * args[0]}) & 255 : 0; }}"
        if op == "position_quotient":
            return f"{{ int64_t s = L - 1 + {args[1]}; {out} = s >= 0 ? s / {args[0]} : -1; }}"
        if op == "read_at":
            return f"if ({a[0]} < 0 || {a[0]} >= L) FAIL(2); {out} = tokens[{a[0]}];"
        if op == "add":
            return f"{out} = {a[0]} + {a[1]};"
        if op == "scale":
            return f"{out} = {a[0]} * {a[1]};"
        if op == "relu":
            return f"{out} = {a[0]} > 0 ? {a[0]} : 0;"
        if op == "clip":
            return f"{out} = {a[0]} < {a[1]} ? {a[1]} : {a[0]} > {a[2]} ? {a[2]} : {a[0]};"
        if op == "equals":
            return f"{out} = {a[0]} == {a[1]};"
        if op == "mod":
            return f"if ({a[0]} < 0) FAIL(5); {out} = {a[0]} % {a[1]};"
        if op == "select":
            return f"if ({a[0]} != 0 && {a[0]} != 1) FAIL(3); {out} = {a[0]} ? {a[1]} : {a[2]};"
        if op == "lookup":
            x, lower, table = self._value(args[0]), args[1], args[2]
            name = f"T{len(tables)}"
            tables.append(f"static const int64_t {name}[{len(table)}] = {{{', '.join(map(str, table))}}};")
            return (
                f"{{ int64_t k = {x} - {_literal(lower)}; if (k < 0 || k >= {len(table)}) FAIL(4); "
                f"{out} = {name}[k]; }}"
            )
        if op == "match":
            m = self.matches.index(node)
            digits, period, phase, start, sentinel, end = args[:6]
            query = [self._value(q) for q in args[6 : 6 + digits]]
            keys = [self._value(k) for k in args[6 + 2 * digits : 6 + 3 * digits]]
            pack = lambda values: " | ".join(f"((uint64_t)({x}) & 0xffff) << {16 * j}" for j, x in enumerate(values))
            in_range = " && ".join(f"{x} >= 0 && {x} <= 0xffff" for x in query)
            tagged = f"p >= {start} && p % {period} == {phase}" + (f" && p < {end}" if end != -1 else "")
            return (
                f"{{ int64_t p = L - 1; if ({tagged}) {{ if (map_put(&maps[{m}], {pack(keys)}, p)) FAIL(9); }}\n"
                f"      int64_t w = ({in_range}) ? map_get(&maps[{m}], {pack(query)}) : -1;\n"
                f"      if (w < 0) {{ if ({sentinel} < {start} || {sentinel} >= L) FAIL(6); w = {sentinel}; }}\n"
                f"      {out} = w; }}"
            )
        if op == "latest":
            m = self.latests.index(node)
            _, period, phase, start = args
            return f"{{ int64_t w = latest_query({m}, {period}, {phase}, {start}, {a[0]}, tokens, L); if (w < 0) FAIL(7); {out} = w; }}"
        raise ValueError(f"unknown operation: {op}")

    def _source(self, order, root):
        machine, tables, body = self.machine, [], []
        for node in order:
            body.append("    " + self._node(node, tables))
        # Prefill passes compute only what match keys need at earlier tagged positions.
        key_nodes = set()
        for n in self.matches:
            digits = n.args[0]
            for k in n.args[6 + 2 * digits : 6 + 3 * digits]:
                if isinstance(k, Expr):
                    key_nodes.update(nodes(k))
        prefill = []
        for node in order:
            if node in key_nodes:
                prefill.append("    " + self._node(node, tables))
        for n in self.matches:
            m = self.matches.index(n)
            digits, period, phase, start, _, end = n.args[:6]
            keys = [self._value(k) for k in n.args[6 + 2 * digits : 6 + 3 * digits]]
            pack = " | ".join(f"((uint64_t)({x}) & 0xffff) << {16 * j}" for j, x in enumerate(keys))
            tagged = f"p >= {start} && p % {period} == {phase}" + (f" && p < {end}" if end != -1 else "")
            prefill.append(
                f"    {{ int64_t p = L - 1; if ({tagged}) {{ if (map_put(&maps[{m}], {pack}, p)) FAIL(9); }} }}"
            )
        vocab = machine.upper - machine.lower + 1
        halts = " || ".join(f"t == {h}" for h in machine.halt_values) or "0"
        n_maps, n_latest = max(1, len(self.matches)), max(1, len(self.latests))
        latest_info = ", ".join(f"{{{n.args[1]}, {n.args[2]}, {n.args[3]}}}" for n in self.latests) or "{1, 0, 0}"
        return (
            _PRELUDE
            + "\n".join(tables)
            + f"""

static Map maps[{n_maps}];
static int32_t *newest[{n_latest}];              /* Per latest node: newest tagged position by token value. */
static int64_t indexed[{n_latest}];              /* Per latest node: positions indexed so far. */
static const int64_t latest_tag[{n_latest}][3] = {{{latest_info}}};
static int64_t v[{len(order)}];

/* Nearest tagged token value to key, then the newest position (reference.py). */
static int64_t latest_query(int m, int64_t period, int64_t phase, int64_t start, int64_t key,
                            const int64_t *tokens, int64_t L) {{
    for (; indexed[m] < L; indexed[m]++) {{
        int64_t p = indexed[m];
        if (p >= start && p % period == phase) newest[m][tokens[p] - ({machine.lower})] = (int32_t)p;
    }}
    int64_t k = key - ({machine.lower});
    for (int64_t d = 0; d <= {vocab}; d++) {{
        int64_t a = k - d >= 0 && k - d < {vocab} ? newest[m][k - d] : -1;
        int64_t b = d && k + d >= 0 && k + d < {vocab} ? newest[m][k + d] : -1;
        if (a >= 0 || b >= 0) return a > b ? a : b;
    }}
    return -1;
}}

static int64_t pass(const int64_t *tokens, int64_t L, int64_t *error) {{
{chr(10).join(body)}
    return v[{self.index[root]}];
}}

static int64_t prefill(const int64_t *tokens, int64_t L, int64_t *error) {{
{chr(10).join(prefill)}
    return 0;
}}

static int tagged_for_match(int64_t p) {{
    {"".join(f"if (p >= {n.args[3]} && p % {n.args[1]} == {n.args[2]}" + (f" && p < {n.args[5]}" if n.args[5] != -1 else "") + ") return 1; " for n in self.matches)}
    return 0;
}}

/* Extend tokens[0:length] (capacity entries) until a halt token, an error, or
   the capacity; returns the new length and sets *error (0 when none). */
int64_t draft_run(int64_t *tokens, int64_t length, int64_t capacity, int64_t *error) {{
    *error = 0;
    for (int m = 0; m < {n_maps}; m++) {{ free(maps[m].keys); free(maps[m].positions); memset(&maps[m], 0, sizeof(Map)); }}
    for (int m = 0; m < {len(self.latests)}; m++) {{
        free(newest[m]);
        newest[m] = malloc(sizeof(int32_t) * {vocab});
        if (!newest[m]) {{ *error = 9; return length; }}
        for (int64_t i = 0; i < {vocab}; i++) newest[m][i] = -1;
        indexed[m] = 0;
    }}
    for (int64_t p = 0; p < length - 1; p++) {{
        if (!tagged_for_match(p)) continue;
        prefill(tokens, p + 1, error);
        if (*error) return length;
    }}
    while (length < capacity) {{
        int64_t t = tokens[length - 1];
        if ({halts}) break;
        int64_t next = pass(tokens, length, error);
        if (*error) break;
        if (next < {machine.lower} || next > {machine.upper}) {{ *error = 8; break; }}
        tokens[length++] = next;
    }}
    return length;
}}
"""
        )

    # -- Running ---------------------------------------------------------------

    def run(self, prefix, max_steps=100):
        """The prefix plus up to max_steps tokens of the source program (stopping
        after a halt token), as reference.run would produce, or fewer if it fails."""
        tokens = np.zeros(len(prefix) + max_steps, dtype=np.int64)
        tokens[: len(prefix)] = prefix
        error = ctypes.c_int64(0)
        length = self._run(
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(prefix), len(tokens), ctypes.byref(error)
        )
        self.error = ERRORS.get(error.value) if error.value else None
        return tokens[:length].tolist()


def _build(source):
    """Compile C source once per distinct program into the user's cache."""
    command = ["cc", "-O2", "-shared", "-fPIC"]
    digest = hashlib.sha256(" ".join(command).encode() + source.encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "firebrand" / "drafts"
    library = cache / f"draft_{digest}.so"
    if not library.exists():
        cache.mkdir(parents=True, exist_ok=True)
        c_file = cache / f"draft_{digest}.c"
        c_file.write_text(source)
        partial = library.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(command + ["-o", str(partial), str(c_file)], check=True)
        partial.replace(library)
    return ctypes.CDLL(str(library))
