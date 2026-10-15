"""Summarize an rv32emu run: per-frame costs, instruction mix, and hot functions.

usage: profile_report.py RUN_DIR ELF MAP [--skip N] > report.json

RUN_DIR holds summary.json, frames.csv, and profile.bin from rv32emu. Functions
come from `nm` symbols; each function's source object comes from the linker map,
which groups it into a subsystem (renderer, game simulation, C library, ...).
"""

import bisect
import csv
import json
import re
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NM = Path(__file__).resolve().parents[2] / "third_party/rv32im-gcc/bin/riscv32-none-elf-nm"

GROUPS = [
    ("renderer", r"^(r_\w+|v_video|i_video|f_wipe|i_scale)\.o$"),
    ("fixed-point math", r"^m_fixed\.o$"),
    ("game simulation", r"^(p_\w+|g_game|d_\w+|m_random|info|doomstat|tables|i_timer|i_input)\.o$"),
    ("HUD, menus, screens", r"^(hu_\w+|st_\w+|wi_stuff|am_map|m_menu|f_finale|m_cheat)\.o$"),
    ("WAD and zone memory", r"^(w_\w+|z_zone|memio|sha1|m_misc|m_config|m_argv)\.o$"),
    ("platform and startup", r"^(doomgeneric\w*|syscalls|crt0|i_system|i_sound|s_sound|dummy)\.o$"),
    ("C library", r"^(libc|libm)\.a"),
    ("compiler runtime (libgcc)", r"^libgcc\.a"),
]


def group_of(obj):
    name = obj.rsplit("/", 1)[-1]
    for label, pattern in GROUPS:
        if re.search(pattern, name):
            return label
    return "other"


def text_sections(map_path):
    """(address, size, object) for each input .text section in the linker map."""
    sections, pending = [], None
    line_re = re.compile(r"^\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(\S+)")
    for line in Path(map_path).read_text().splitlines():
        if re.match(r"^ \.text[.\s]", line) or line.startswith(" .text."):
            rest = line.split(None, 1)
            if len(rest) == 2 and line_re.match(" " + rest[1]):
                m = line_re.match(" " + rest[1])
                sections.append((int(m[1], 16), int(m[2], 16), m[3]))
                pending = None
            else:
                pending = True
            continue
        if pending:
            m = line_re.match(line)
            if m:
                sections.append((int(m[1], 16), int(m[2], 16), m[3]))
            pending = None
    return sorted(s for s in sections if s[1])


def functions(elf):
    out = subprocess.run([str(NM), "-S", "--defined-only", str(elf)], capture_output=True,
                         text=True, check=True).stdout
    symbols = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "Tt":
            symbols.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
    return sorted(symbols)


def main():
    run, elf, map_path = (Path(a) for a in sys.argv[1:4])
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    summary = json.loads((run / "summary.json").read_text())
    rows = list(csv.DictReader(open(run / "frames.csv")))
    frames = [{k: (int(v) if k != "checksum" else v) for k, v in row.items()} for row in rows]

    data = (run / "profile.bin").read_bytes()
    base = struct.unpack_from("<I", data)[0]
    counts = struct.unpack_from(f"<{(len(data) - 4) // 8}Q", data, 4)
    symbols, sections = functions(elf), text_sections(map_path)
    starts = [s[0] for s in symbols]
    section_starts = [s[0] for s in sections]
    by_function, by_group = defaultdict(int), defaultdict(int)
    function_group = {}
    for i, count in enumerate(counts):
        if not count:
            continue
        pc = base + 4 * i
        k = bisect.bisect_right(starts, pc) - 1
        name = symbols[k][2] if k >= 0 and pc < symbols[k][0] + max(symbols[k][1], 4) else "?"
        j = bisect.bisect_right(section_starts, pc) - 1
        obj = sections[j][2] if j >= 0 and pc < sections[j][0] + sections[j][1] else "?"
        by_function[name] += count
        function_group[name] = group_of(obj)
        by_group[function_group[name]] += count
    total = sum(counts)

    measured = frames[skip:]
    def stats(key):
        values = sorted(f[key] for f in measured)
        return {"mean": statistics.fmean(values), "median": statistics.median(values),
                "p95": values[int(0.95 * (len(values) - 1))], "max": values[-1], "min": values[0]}

    ops = summary["ops"]
    classes = {
        "ALU (add/sub/logic/compare/lui/auipc)": ["add", "addi", "sub", "and", "andi", "or", "ori", "xor", "xori",
                                                  "slt", "slti", "sltu", "sltiu", "lui", "auipc"],
        "shifts": ["sll", "slli", "srl", "srli", "sra", "srai"],
        "loads": ["lb", "lh", "lw", "lbu", "lhu"],
        "stores": ["sb", "sh", "sw"],
        "branches": ["beq", "bne", "blt", "bge", "bltu", "bgeu"],
        "jumps": ["jal", "jalr"],
        "multiply": ["mul", "mulh", "mulhsu", "mulhu"],
        "divide": ["div", "divu", "rem", "remu"],
        "system": ["ecall", "ebreak", "fence"],
    }
    report = {
        "setup_instructions": sum(f["instructions"] for f in frames[:skip]) if skip else frames[0]["instructions"],
        "frames_measured": len(measured),
        "per_frame": {key: stats(key) for key in ("instructions", "loads", "stores", "multiplies", "divides",
                                                  "read_pages", "written_pages", "read_words", "written_words")},
        "total_instructions": summary["instructions"],
        "op_mix": {label: sum(ops[o] for o in names) / summary["instructions"] for label, names in classes.items()},
        "ops": ops,
        "groups": {k: v / total for k, v in sorted(by_group.items(), key=lambda kv: -kv[1])},
        "top_functions": [{"name": n, "share": c / total, "group": function_group[n]}
                          for n, c in sorted(by_function.items(), key=lambda kv: -kv[1])[:25]],
        "text_bytes": summary["text_bytes"],
        "pages_read_ever": summary["pages_read_ever"],
        "pages_written_ever": summary["pages_written_ever"],
        "exit_code": summary["exit_code"],
    }
    json.dump(report, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
