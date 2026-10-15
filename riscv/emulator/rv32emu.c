/*
 * rv32emu: an RV32IM reference emulator with execution statistics.
 *
 * Guest memory is RAM at [0, RAM_SIZE) plus an optional read-only image (the
 * WAD) at ROM_BASE. The host interface is ECALL:
 *
 *   a7 = 93       exit(a0)
 *   a7 = 64       write(a0 = fd, a1 = buffer, a2 = length) -> length
 *   a7 = 0x10000  frame(a0 = 8-bit framebuffer, a1 = 256 x {b,g,r,a} palette)
 *   a7 = 0x10001  ticks() -> milliseconds on a deterministic virtual clock
 *   a7 = 0x10002  sleep(a0 = milliseconds): advances the virtual clock
 *   a7 = 0x10003  key() -> (pressed << 8 | key) + 1 for the next scripted event, or 0
 *   a7 = 0x10004  args(a0 = buffer, a1 = capacity) -> bytes of NUL-separated argv
 *   a7 = 0x10005  rom() -> ROM base address; a0 of the next call is its size
 *
 * The clock advances one DOOM tic (1000/35 ms) per frame plus sleeps, so a run is
 * a pure function of the program, ROM, and arguments.
 *
 * --context-records C follows how the transformer interpreter fills contexts of
 * C records (an instruction is one record, a division 65; an instruction that
 * does not fit restarts in the next context). --save-states / --load-state save
 * and resume machine states at context boundaries.
 *
 * --draft FILE writes the tokens the interpreter generates, per context: a
 * uint64 record count, the ten-byte records (including division micro-steps),
 * then a byte that is 1 if the context continues, followed by the state there
 * (pc, 32 registers, the first --state-bytes of RAM). The exit call writes no record.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAM_SIZE (64u << 20)
#define ROM_BASE 0x40000000u
#define PAGE_BITS 12

enum {
    OP_LUI, OP_AUIPC, OP_JAL, OP_JALR,
    OP_BEQ, OP_BNE, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU,
    OP_LB, OP_LH, OP_LW, OP_LBU, OP_LHU, OP_SB, OP_SH, OP_SW,
    OP_ADDI, OP_SLTI, OP_SLTIU, OP_XORI, OP_ORI, OP_ANDI, OP_SLLI, OP_SRLI, OP_SRAI,
    OP_ADD, OP_SUB, OP_SLL, OP_SLT, OP_SLTU, OP_XOR, OP_SRL, OP_SRA, OP_OR, OP_AND,
    OP_MUL, OP_MULH, OP_MULHSU, OP_MULHU, OP_DIV, OP_DIVU, OP_REM, OP_REMU,
    OP_FENCE, OP_ECALL, OP_EBREAK, OP_COUNT
};

static const char *op_names[OP_COUNT] = {
    "lui", "auipc", "jal", "jalr", "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw",
    "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai",
    "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    "fence", "ecall", "ebreak",
};

typedef struct {
    uint64_t instructions, loads, stores, load_bytes, store_bytes, misaligned;
    uint64_t taken_branches, register_writes;
    uint64_t ops[OP_COUNT];
    uint64_t read_pages, written_pages, read_words, written_words;
} Counters;

static uint8_t *ram;
static uint8_t *rom;
static uint32_t rom_size;
static uint32_t regs[32];
static uint32_t pc;

static Counters total, frame;
static uint32_t generation = 1;          /* Per-frame stamp for working sets. */
static uint32_t *word_read_stamp, *word_write_stamp;
static uint32_t *page_read_stamp, *page_write_stamp;
static uint64_t *pc_counts;              /* Instruction count by word address. */
static uint32_t text_start, text_end;

static uint64_t clock_us;                /* Virtual time in microseconds. */
static int frames, max_frames = -1, skip_frames = 0;
static uint64_t max_instructions = UINT64_MAX;
static FILE *frame_log;
static const char *dump_dir;
static int dump_every = 0;
static char args_blob[4096];
static uint32_t args_length;
static int key_events[256][2], key_count, key_next;
static int key_frames[256];
static int exit_code, exited;
static FILE *trace_file;                 /* One line per instruction: see trace_record(). */
static uint64_t context_records;         /* Records per context, or 0 when not tracking. */
static uint64_t context_used, contexts;  /* Records used in the current context; boundaries so far. */
static const char *state_dir;
static int save_every = 1;
static uint32_t state_bytes;
static int merged_output;                /* Guest stderr goes to stdout, in order. */
static FILE *boundary_log;               /* One line per boundary: context, instructions. */
static FILE *draft_file;                 /* Transformer tokens per context: see the header. */
static uint8_t *draft;                   /* The current context's records, ten bytes each. */
static uint64_t draft_records;

static void fail(const char *message, uint32_t value) {
    fprintf(stderr, "rv32emu: %s 0x%08" PRIx32 " at pc 0x%08" PRIx32 " after %" PRIu64 " instructions\n",
            message, value, pc, total.instructions + frame.instructions);
    exit(2);
}

static void track(uint32_t address, uint32_t size, int write) {
    if (address >= RAM_SIZE) return;
    uint32_t *words = write ? word_write_stamp : word_read_stamp;
    uint32_t *pages = write ? page_write_stamp : page_read_stamp;
    for (uint32_t word = address >> 2; word <= (address + size - 1) >> 2; word++) {
        if (words[word] != generation) {
            words[word] = generation;
            if (write) frame.written_words++; else frame.read_words++;
        }
    }
    uint32_t page = address >> PAGE_BITS;
    if (pages[page] != generation) {
        pages[page] = generation;
        if (write) frame.written_pages++; else frame.read_pages++;
    }
}

static uint8_t *locate(uint32_t address, uint32_t size, int write) {
    if (address + size <= RAM_SIZE && address + size >= address) return ram + address;
    if (!write && address >= ROM_BASE && address - ROM_BASE + size <= rom_size) return rom + (address - ROM_BASE);
    fail(write ? "store outside writable memory" : "load outside memory", address);
    return NULL;
}

static uint32_t load(uint32_t address, uint32_t size) {
    uint8_t *p = locate(address, size, 0);
    uint32_t value = 0;
    for (uint32_t i = 0; i < size; i++) value |= (uint32_t)p[i] << (8 * i);
    frame.loads++;
    frame.load_bytes += size;
    if (address & (size - 1)) frame.misaligned++;
    track(address, size, 0);
    return value;
}

static void store(uint32_t address, uint32_t size, uint32_t value) {
    uint8_t *p = locate(address, size, 1);
    for (uint32_t i = 0; i < size; i++) p[i] = (uint8_t)(value >> (8 * i));
    frame.stores++;
    frame.store_bytes += size;
    if (address & (size - 1)) frame.misaligned++;
    track(address, size, 1);
}

static void add_counters(Counters *into, const Counters *from) {
    uint64_t *a = (uint64_t *)into;
    const uint64_t *b = (const uint64_t *)from;
    for (size_t i = 0; i < sizeof(Counters) / sizeof(uint64_t); i++) a[i] += b[i];
}

static uint64_t fnv1a(const uint8_t *data, size_t length) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; i++) hash = (hash ^ data[i]) * 1099511628211ull;
    return hash;
}

static void dump_frame(const uint8_t *pixels, uint32_t palette_address) {
    char path[512];
    snprintf(path, sizeof path, "%s/frame_%05d.ppm", dump_dir, frames);
    FILE *out = fopen(path, "wb");
    if (!out) fail("cannot write frame dump", 0);
    fprintf(out, "P6\n320 200\n255\n");
    for (int i = 0; i < 320 * 200; i++) {
        const uint8_t *c = locate(palette_address + 4u * pixels[i], 4, 0);
        uint8_t rgb[3] = {c[2], c[1], c[0]};
        fwrite(rgb, 1, 3, out);
    }
    fclose(out);
}

/* A frame closes the current measurement interval. */
static void end_frame(uint32_t framebuffer, uint32_t palette) {
    const uint8_t *pixels = locate(framebuffer, 320 * 200, 0);
    uint64_t checksum = fnv1a(pixels, 320 * 200);
    if (frame_log) {
        fprintf(frame_log, "%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%016" PRIx64 "\n",
                frames, frame.instructions, frame.loads, frame.stores, frame.taken_branches,
                frame.ops[OP_MUL] + frame.ops[OP_MULH] + frame.ops[OP_MULHSU] + frame.ops[OP_MULHU],
                frame.ops[OP_DIV] + frame.ops[OP_DIVU] + frame.ops[OP_REM] + frame.ops[OP_REMU],
                frame.read_pages, frame.written_pages, frame.read_words, frame.written_words, checksum);
    }
    if (dump_dir && dump_every > 0 && frames % dump_every == 0) dump_frame(pixels, palette);
    add_counters(&total, &frame);
    memset(&frame, 0, sizeof frame);
    generation++;
    frames++;
    clock_us += 1000000 / 35;
    if (frames == skip_frames) {
        /* Totals and the profile describe only the frames after this point. */
        memset(&total, 0, sizeof total);
        if (pc_counts) memset(pc_counts, 0, ((text_end - text_start) / 4 + 1) * sizeof *pc_counts);
    }
}

static void ecall(void) {
    uint32_t number = regs[17], a0 = regs[10], a1 = regs[11], a2 = regs[12];
    switch (number) {
    case 93:
        exit_code = (int32_t)a0;
        exited = 1;
        break;
    case 64:
        if (a0 == 1 || a0 == 2) fwrite(locate(a1, a2, 0), 1, a2, a0 == 1 || merged_output ? stdout : stderr);
        regs[10] = a2;
        break;
    case 0x10000:
        end_frame(a0, a1);
        break;
    case 0x10001:
        regs[10] = (uint32_t)(clock_us / 1000);
        break;
    case 0x10002:
        clock_us += (uint64_t)a0 * 1000;
        break;
    case 0x10003:
        regs[10] = 0;
        if (key_next < key_count && frames >= key_frames[key_next]) {
            regs[10] = (uint32_t)((key_events[key_next][0] << 8 | key_events[key_next][1]) + 1);
            key_next++;
        }
        break;
    case 0x10004: {
        uint32_t n = args_length < a1 ? args_length : a1;
        memcpy(locate(a0, n ? n : 1, 1), args_blob, n);
        regs[10] = n;
        break;
    }
    case 0x10005:
        regs[10] = ROM_BASE;
        regs[11] = rom_size;
        break;
    default:
        fail("unknown ecall", number);
    }
}

static inline int32_t imm_i(uint32_t in) { return (int32_t)in >> 20; }
static inline int32_t imm_s(uint32_t in) { return ((int32_t)in >> 25 << 5) | ((in >> 7) & 31); }
static inline int32_t imm_b(uint32_t in) {
    return ((int32_t)in >> 31 << 12) | ((in >> 7 & 1) << 11) | ((in >> 25 & 63) << 5) | ((in >> 8 & 15) << 1);
}
static inline int32_t imm_j(uint32_t in) {
    return ((int32_t)in >> 31 << 20) | (in & 0xff000) | ((in >> 20 & 1) << 11) | ((in >> 21 & 1023) << 1);
}

/*
 * The per-instruction record the transformer interpreter emits: the next pc as a
 * word index, the written value, and a key naming what was written (0 = nothing,
 * 1..31 = register x1..x31, 96 + w = memory word w, holding the whole merged word;
 * 32..95 are the interpreter's division scratch keys, never written here).
 */
static void trace_record(uint32_t next, uint32_t rd, int wrote_register, int32_t store_address) {
    uint32_t key = 0, value = 0;
    if (store_address >= 0) {
        uint32_t w = (uint32_t)store_address >> 2;
        key = 96 + w;
        memcpy(&value, ram + 4 * w, 4);
    } else if (wrote_register && rd) {
        key = rd;
        value = regs[rd];
    }
    fprintf(trace_file, "%" PRIu32 " %" PRIu32 " %" PRIu32 "\n", next >> 2, key, value);
}

/* The key and value of an instruction's record (see trace_record). */
static void record_write(uint32_t rd, int wrote_register, int32_t store_address, uint32_t *key, uint32_t *value) {
    *key = *value = 0;
    if (store_address >= 0) {
        uint32_t w = (uint32_t)store_address >> 2;
        *key = 96 + w;
        memcpy(value, ram + 4 * w, 4);
    } else if (wrote_register && rd) {
        *key = rd;
        *value = regs[rd];
    }
}

/* One draft record: [next pc word: 3 bytes][value: 4 bytes][key: 3 bytes]. */
static void draft_put(uint32_t next_word, uint32_t key, uint32_t value) {
    uint8_t *r = draft + 10 * draft_records++;
    for (int k = 0; k < 3; k++) r[k] = next_word >> 8 * k & 255;
    for (int k = 0; k < 4; k++) r[3 + k] = value >> 8 * k & 255;
    for (int k = 0; k < 3; k++) r[7 + k] = key >> 8 * k & 255;
}

/* The first `count` of a division's 64 micro-step records, as the interpreter
   computes them (rv32.py, _division): restoring division of the magnitudes,
   alternately writing the partial remainder A(i) under key 32 + 2i and the
   quotient Q(i) under key 33 + 2i, at the division's own pc. */
static void draft_division(uint32_t at, uint32_t a, uint32_t b, uint32_t f3, int count) {
    int is_signed = f3 == 4 || f3 == 6;
    uint32_t dividend = is_signed && a >> 31 ? -a : a, divisor = is_signed && b >> 31 ? -b : b;
    uint32_t remainder = 0, quotient = dividend;
    for (int i = 0; i < 32 && 2 * i < count; i++) {
        uint64_t shifted = (uint64_t)remainder << 1 | quotient >> 31;
        uint32_t fits = shifted >= divisor;
        remainder = (uint32_t)(fits ? shifted - divisor : shifted);
        quotient = quotient << 1 | fits;
        draft_put(at >> 2, 32 + 2 * i, remainder);
        if (2 * i + 1 < count) draft_put(at >> 2, 33 + 2 * i, quotient);
    }
}

/* End the current context's draft, with the machine state if it continues. */
static void draft_flush(int continues) {
    uint8_t flag = (uint8_t)continues;
    fwrite(&draft_records, 8, 1, draft_file);
    fwrite(draft, 10, draft_records, draft_file);
    fwrite(&flag, 1, 1, draft_file);
    if (continues) {
        fwrite(&pc, 4, 1, draft_file);
        fwrite(regs, 4, 32, draft_file);
        fwrite(ram, 1, state_bytes, draft_file);
    }
    if (ferror(draft_file)) fail("cannot write draft", 0);
    draft_records = 0;
}

#define STATE_MAGIC 0x31767273u           /* "srv1" */

static void save_state(void) {
    for (uint32_t page = state_bytes >> PAGE_BITS; page < RAM_SIZE >> PAGE_BITS; page++)
        if (page_write_stamp[page]) fail("state would drop a write above --state-bytes", page << PAGE_BITS);
    char path[512];
    snprintf(path, sizeof path, "%s/state_%06" PRIu64 ".bin", state_dir, contexts);
    FILE *out = fopen(path, "wb");
    if (!out) fail("cannot write state", 0);
    uint32_t header[2] = {STATE_MAGIC, state_bytes};
    uint64_t counters[4] = {total.instructions + frame.instructions, contexts, clock_us, (uint64_t)frames};
    int32_t keys = key_next;
    fwrite(header, 4, 2, out);
    fwrite(counters, 8, 4, out);
    fwrite(&keys, 4, 1, out);
    fwrite(&pc, 4, 1, out);
    fwrite(regs, 4, 32, out);
    fwrite(ram, 1, state_bytes, out);
    if (fclose(out)) fail("cannot write state", 0);
}

static void load_state(const char *path) {
    FILE *in = fopen(path, "rb");
    if (!in) { perror(path); exit(1); }
    uint32_t header[2];
    uint64_t counters[4];
    int32_t keys;
    if (fread(header, 4, 2, in) != 2 || header[0] != STATE_MAGIC || header[1] > RAM_SIZE
        || fread(counters, 8, 4, in) != 4 || fread(&keys, 4, 1, in) != 1 || fread(&pc, 4, 1, in) != 1
        || fread(regs, 4, 32, in) != 32) {
        fprintf(stderr, "%s: not an rv32emu state\n", path);
        exit(1);
    }
    memset(ram, 0, RAM_SIZE);
    if (fread(ram, 1, header[1], in) != header[1]) { fprintf(stderr, "%s: truncated\n", path); exit(1); }
    fclose(in);
    total.instructions = counters[0];
    contexts = counters[1];
    clock_us = counters[2];
    frames = (int)counters[3];
    key_next = keys;
}

/* Called before each instruction: does it start a new context? */
static void context_step(uint32_t in) {
    uint64_t records = (in & 127) == 0x33 && in >> 25 == 1 && (in >> 12 & 7) >= 4 ? 65 : 1;
    if (context_used + records <= context_records) {
        context_used += records;
        return;
    }
    if (draft_file) {
        if (records == 65)                    /* A division cut short; it restarts next context. */
            draft_division(pc, regs[in >> 15 & 31], regs[in >> 20 & 31], in >> 12 & 7,
                           (int)(context_records - context_used));
        draft_flush(1);
    }
    contexts++;
    if (boundary_log)
        fprintf(boundary_log, "%" PRIu64 " %" PRIu64 "\n", contexts, total.instructions + frame.instructions);
    if (state_dir && contexts % save_every == 0) save_state();
    context_used = records;
}

static void run(void) {
    while (!exited) {
        if (total.instructions + frame.instructions >= max_instructions) break;
        if (max_frames >= 0 && frames >= max_frames) break;
        if (pc & 3) fail("misaligned pc", pc);
        if (pc < text_start || pc >= text_end) fail("fetch outside executable segment", pc);
        uint32_t in;
        memcpy(&in, ram + pc, 4);                      /* Fetches are not data loads. */
        if (context_records) context_step(in);
        if (pc_counts && pc >= text_start && pc < text_end) pc_counts[(pc - text_start) >> 2]++;
        uint32_t rd = in >> 7 & 31, rs1 = in >> 15 & 31, rs2 = in >> 20 & 31, f3 = in >> 12 & 7, f7 = in >> 25;
        uint32_t a = regs[rs1], b = regs[rs2], next = pc + 4, result = 0;
        int op = -1, writes = 1;
        int32_t store_address = -1;
        switch (in & 127) {
        case 0x37: op = OP_LUI; result = in & 0xfffff000; break;
        case 0x17: op = OP_AUIPC; result = pc + (in & 0xfffff000); break;
        case 0x6f: op = OP_JAL; result = next; next = pc + imm_j(in); break;
        case 0x67: op = OP_JALR; result = next; next = (a + imm_i(in)) & ~1u; break;
        case 0x63: {
            static const int ops[8] = {OP_BEQ, OP_BNE, -1, -1, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU};
            int taken;
            op = ops[f3];
            writes = 0;
            switch (f3) {
            case 0: taken = a == b; break;
            case 1: taken = a != b; break;
            case 4: taken = (int32_t)a < (int32_t)b; break;
            case 5: taken = (int32_t)a >= (int32_t)b; break;
            case 6: taken = a < b; break;
            case 7: taken = a >= b; break;
            default: fail("illegal branch", in); taken = 0;
            }
            if (taken) { next = pc + imm_b(in); frame.taken_branches++; }
            break;
        }
        case 0x03: {
            uint32_t address = a + imm_i(in);
            switch (f3) {
            case 0: op = OP_LB; result = (uint32_t)(int32_t)(int8_t)load(address, 1); break;
            case 1: op = OP_LH; result = (uint32_t)(int32_t)(int16_t)load(address, 2); break;
            case 2: op = OP_LW; result = load(address, 4); break;
            case 4: op = OP_LBU; result = load(address, 1); break;
            case 5: op = OP_LHU; result = load(address, 2); break;
            default: fail("illegal load", in);
            }
            break;
        }
        case 0x23: {
            uint32_t address = a + imm_s(in);
            writes = 0;
            store_address = (int32_t)address;
            switch (f3) {
            case 0: op = OP_SB; store(address, 1, b); break;
            case 1: op = OP_SH; store(address, 2, b); break;
            case 2: op = OP_SW; store(address, 4, b); break;
            default: fail("illegal store", in);
            }
            break;
        }
        case 0x13: {
            int32_t imm = imm_i(in);
            switch (f3) {
            case 0: op = OP_ADDI; result = a + imm; break;
            case 2: op = OP_SLTI; result = (int32_t)a < imm; break;
            case 3: op = OP_SLTIU; result = a < (uint32_t)imm; break;
            case 4: op = OP_XORI; result = a ^ imm; break;
            case 6: op = OP_ORI; result = a | imm; break;
            case 7: op = OP_ANDI; result = a & imm; break;
            case 1: if (f7) fail("illegal shift", in); op = OP_SLLI; result = a << rs2; break;
            case 5:
                if (f7 == 0) { op = OP_SRLI; result = a >> rs2; }
                else if (f7 == 0x20) { op = OP_SRAI; result = (uint32_t)((int32_t)a >> rs2); }
                else fail("illegal shift", in);
                break;
            }
            break;
        }
        case 0x33:
            if (f7 == 1) {
                int64_t sa = (int32_t)a, sb = (int32_t)b;
                switch (f3) {
                case 0: op = OP_MUL; result = a * b; break;
                case 1: op = OP_MULH; result = (uint32_t)((uint64_t)(sa * sb) >> 32); break;
                case 2: op = OP_MULHSU; result = (uint32_t)((uint64_t)(sa * (int64_t)(uint64_t)b) >> 32); break;
                case 3: op = OP_MULHU; result = (uint32_t)(((uint64_t)a * b) >> 32); break;
                case 4: op = OP_DIV;
                    result = b == 0 ? 0xffffffffu : (a == 0x80000000u && b == 0xffffffffu) ? a
                           : (uint32_t)((int32_t)a / (int32_t)b);
                    break;
                case 5: op = OP_DIVU; result = b == 0 ? 0xffffffffu : a / b; break;
                case 6: op = OP_REM;
                    result = b == 0 ? a : (a == 0x80000000u && b == 0xffffffffu) ? 0
                           : (uint32_t)((int32_t)a % (int32_t)b);
                    break;
                case 7: op = OP_REMU; result = b == 0 ? a : a % b; break;
                }
            } else if (f7 == 0 || f7 == 0x20) {
                int alt = f7 == 0x20;
                switch (f3) {
                case 0: op = alt ? OP_SUB : OP_ADD; result = alt ? a - b : a + b; break;
                case 1: op = OP_SLL; result = a << (b & 31); break;
                case 2: op = OP_SLT; result = (int32_t)a < (int32_t)b; break;
                case 3: op = OP_SLTU; result = a < b; break;
                case 4: op = OP_XOR; result = a ^ b; break;
                case 5: op = alt ? OP_SRA : OP_SRL;
                    result = alt ? (uint32_t)((int32_t)a >> (b & 31)) : a >> (b & 31);
                    break;
                case 6: op = OP_OR; result = a | b; break;
                case 7: op = OP_AND; result = a & b; break;
                }
                if (alt && f3 != 0 && f3 != 5) fail("illegal op", in);
            } else fail("illegal op", in);
            break;
        case 0x0f: op = OP_FENCE; writes = 0; break;
        case 0x73:
            writes = 0;
            if (in == 0x00000073) {
                op = OP_ECALL;
                frame.ops[op]++;
                frame.instructions++;
                uint32_t call = regs[17];
                pc = next;
                ecall();
                /* write returns its length in a0; other calls write no register. */
                if (trace_file) trace_record(next, 10, call == 64, -1);
                if (draft_file && !exited) draft_put(next >> 2, call == 64 ? 10 : 0, call == 64 ? regs[10] : 0);
                continue;
            }
            if (in == 0x00100073) { op = OP_EBREAK; fail("ebreak", in); }
            fail("unsupported system instruction", in);
            break;
        default:
            fail("illegal instruction", in);
        }
        if (op < 0) fail("illegal instruction", in);
        if (writes && rd) { regs[rd] = result; frame.register_writes++; }
        if (trace_file) trace_record(next, rd, writes, store_address);
        if (draft_file) {
            uint32_t key, value;
            if (op >= OP_DIV && op <= OP_REMU) draft_division(pc, a, b, f3, 64);
            record_write(rd, writes, store_address, &key, &value);
            draft_put(next >> 2, key, value);
        }
        frame.ops[op]++;
        frame.instructions++;
        pc = next;
    }
    add_counters(&total, &frame);
    if (draft_file) draft_flush(0);
}

/* Minimal ELF32 loader: copy PT_LOAD segments and find the entry point. */
static void load_elf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *image = malloc(size);
    if (fread(image, 1, size, f) != (size_t)size) { perror(path); exit(1); }
    fclose(f);
    if (memcmp(image, "\x7f" "ELF", 4) || image[4] != 1 || image[18] != 243) {
        fprintf(stderr, "%s: not an RV32 ELF\n", path);
        exit(1);
    }
    uint32_t entry, phoff;
    uint16_t phentsize, phnum;
    memcpy(&entry, image + 24, 4);
    memcpy(&phoff, image + 28, 4);
    memcpy(&phentsize, image + 42, 2);
    memcpy(&phnum, image + 44, 2);
    text_start = UINT32_MAX;
    text_end = 0;
    for (int i = 0; i < phnum; i++) {
        uint32_t *ph = (uint32_t *)(image + phoff + i * phentsize);
        uint32_t type = ph[0], offset = ph[1], vaddr = ph[2], filesz = ph[4], memsz = ph[5], flags = ph[6];
        if (type != 1) continue;
        if (vaddr + memsz > RAM_SIZE) { fprintf(stderr, "segment outside RAM\n"); exit(1); }
        memcpy(ram + vaddr, image + offset, filesz);
        memset(ram + vaddr + filesz, 0, memsz - filesz);
        if (flags & 1) {                          /* Executable segment. */
            if (vaddr < text_start) text_start = vaddr;
            if (vaddr + memsz > text_end) text_end = vaddr + memsz;
        }
    }
    free(image);
    pc = entry;
}

static uint32_t count_pages(const uint32_t *stamps) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < RAM_SIZE >> PAGE_BITS; i++) count += stamps[i] != 0;
    return count;
}

static void usage(void) {
    fprintf(stderr,
            "usage: rv32emu [options] program.elf [guest args...]\n"
            "  --rom FILE           map FILE read-only at 0x40000000\n"
            "  --frames N           stop after N frames\n"
            "  --max-instructions N stop after N instructions\n"
            "  --frame-log FILE     per-frame CSV statistics\n"
            "  --dump-frames DIR    write PPM frames into DIR\n"
            "  --dump-every K       dump every K-th frame (default 1 with --dump-frames)\n"
            "  --profile FILE       per-PC instruction counts (binary uint64 array)\n"
            "  --summary FILE       JSON totals\n"
            "  --skip-frames N      reset totals and profile after frame N (exclude setup)\n"
            "  --key FRAME:PRESSED:KEY  scripted input event (repeatable)\n"
            "  --trace FILE         per-instruction records: next_pc_word key value\n"
            "  --trace-exit         print the exit code and totals to stderr\n"
            "  --merged-output      write the guest's stderr to stdout too, in order\n"
            "  --context-records C  follow contexts of C transformer records\n"
            "  --boundary-log FILE  per boundary: context number, instructions before it\n"
            "  --save-states DIR    write the state at every K-th boundary (needs --state-bytes)\n"
            "  --save-every K       (default 1)\n"
            "  --state-bytes N      RAM bytes in a saved state\n"
            "  --load-state FILE    resume from a saved state\n"
            "  --draft FILE         the transformer's tokens per context (needs --context-records, --state-bytes)\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *rom_path = NULL, *frame_log_path = NULL, *profile_path = NULL, *summary_path = NULL;
    const char *state_path = NULL;
    int trace_exit = 0, i = 1;
    for (; i < argc && argv[i][0] == '-' && argv[i][1] == '-'; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--trace-exit")) { trace_exit = 1; continue; }
        if (!strcmp(option, "--merged-output")) { merged_output = 1; continue; }
        if (i + 1 >= argc) usage();
        const char *value = argv[++i];
        if (!strcmp(option, "--rom")) rom_path = value;
        else if (!strcmp(option, "--frames")) max_frames = atoi(value);
        else if (!strcmp(option, "--max-instructions")) max_instructions = strtoull(value, NULL, 10);
        else if (!strcmp(option, "--frame-log")) frame_log_path = value;
        else if (!strcmp(option, "--dump-frames")) { dump_dir = value; if (!dump_every) dump_every = 1; }
        else if (!strcmp(option, "--dump-every")) dump_every = atoi(value);
        else if (!strcmp(option, "--profile")) profile_path = value;
        else if (!strcmp(option, "--summary")) summary_path = value;
        else if (!strcmp(option, "--skip-frames")) skip_frames = atoi(value);
        else if (!strcmp(option, "--trace")) trace_file = fopen(value, "w");
        else if (!strcmp(option, "--context-records")) context_records = strtoull(value, NULL, 10);
        else if (!strcmp(option, "--boundary-log")) boundary_log = fopen(value, "w");
        else if (!strcmp(option, "--save-states")) state_dir = value;
        else if (!strcmp(option, "--save-every")) save_every = atoi(value);
        else if (!strcmp(option, "--state-bytes")) state_bytes = (uint32_t)strtoul(value, NULL, 10);
        else if (!strcmp(option, "--load-state")) state_path = value;
        else if (!strcmp(option, "--draft")) {
            draft_file = fopen(value, "wb");
            if (!draft_file) { perror(value); return 1; }
        }
        else if (!strcmp(option, "--key")) {
            if (key_count == 256 || sscanf(value, "%d:%d:%d", &key_frames[key_count],
                                           &key_events[key_count][0], &key_events[key_count][1]) != 3) usage();
            key_count++;
        } else usage();
    }
    if (i >= argc) usage();
    const char *program = argv[i++];
    for (int k = i - 1; k < argc; k++) {                 /* argv[0] is the program name. */
        size_t n = strlen(argv[k]) + 1;
        if (args_length + n > sizeof args_blob) usage();
        memcpy(args_blob + args_length, argv[k], n);
        args_length += n;
    }

    ram = calloc(1, RAM_SIZE);
    word_read_stamp = calloc(RAM_SIZE / 4, 4);
    word_write_stamp = calloc(RAM_SIZE / 4, 4);
    page_read_stamp = calloc(RAM_SIZE >> PAGE_BITS, 4);
    page_write_stamp = calloc(RAM_SIZE >> PAGE_BITS, 4);
    if (!ram || !word_read_stamp || !word_write_stamp) { fprintf(stderr, "out of memory\n"); return 1; }
    load_elf(program);
    if (rom_path) {
        FILE *f = fopen(rom_path, "rb");
        if (!f) { perror(rom_path); return 1; }
        fseek(f, 0, SEEK_END);
        rom_size = (uint32_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        rom = malloc(rom_size);
        if (fread(rom, 1, rom_size, f) != rom_size) { perror(rom_path); return 1; }
        fclose(f);
    }
    if (profile_path) pc_counts = calloc((text_end - text_start) / 4 + 1, sizeof *pc_counts);
    if (frame_log_path) {
        frame_log = fopen(frame_log_path, "w");
        fprintf(frame_log, "frame,instructions,loads,stores,taken_branches,multiplies,divides,"
                           "read_pages,written_pages,read_words,written_words,checksum\n");
    }
    regs[2] = RAM_SIZE - 16;                             /* Default stack; crt0 may override. */
    if (state_dir && (!context_records || !state_bytes || state_bytes > RAM_SIZE || save_every < 1)) usage();
    if (state_path) load_state(state_path);
    if (draft_file) {
        if (!context_records || !state_bytes || state_bytes > RAM_SIZE) usage();
        if (!state_path) regs[2] = 0;          /* The transformer's first prompt has zero registers. */
        draft = malloc(10 * context_records);
        if (!draft) { fprintf(stderr, "out of memory\n"); return 1; }
    }

    run();
    if (draft_file && fclose(draft_file)) fail("cannot write draft", 0);

    if (frame_log) fclose(frame_log);
    if (trace_file) fclose(trace_file);
    if (boundary_log) fclose(boundary_log);
    if (profile_path) {
        FILE *out = fopen(profile_path, "wb");
        fwrite(&text_start, 4, 1, out);
        fwrite(pc_counts, sizeof *pc_counts, (text_end - text_start) / 4, out);
        fclose(out);
    }
    if (summary_path) {
        FILE *out = fopen(summary_path, "w");
        fprintf(out, "{\n  \"exited\": %s,\n  \"exit_code\": %d,\n  \"frames\": %d,\n  \"skipped_frames\": %d,\n  \"instructions\": %" PRIu64
                ",\n  \"loads\": %" PRIu64 ",\n  \"stores\": %" PRIu64 ",\n  \"load_bytes\": %" PRIu64
                ",\n  \"store_bytes\": %" PRIu64 ",\n  \"misaligned\": %" PRIu64
                ",\n  \"taken_branches\": %" PRIu64 ",\n  \"register_writes\": %" PRIu64
                ",\n  \"text_bytes\": %" PRIu32 ",\n  \"pages_read_ever\": %" PRIu32
                ",\n  \"pages_written_ever\": %" PRIu32 ",\n  \"ops\": {",
                exited ? "true" : "false", exit_code, frames, skip_frames, total.instructions, total.loads, total.stores, total.load_bytes,
                total.store_bytes, total.misaligned, total.taken_branches, total.register_writes,
                text_end - text_start, count_pages(page_read_stamp), count_pages(page_write_stamp));
        for (int k = 0; k < OP_COUNT; k++)
            fprintf(out, "%s\"%s\": %" PRIu64, k ? ", " : "", op_names[k], total.ops[k]);
        /* Contiguous runs of 4 KiB pages ever written, as [first, end) byte addresses. */
        fprintf(out, "},\n  \"written_ranges\": [");
        int first = 1;
        for (uint32_t p = 0; p < RAM_SIZE >> PAGE_BITS; p++) {
            if (!page_write_stamp[p] || (p && page_write_stamp[p - 1])) continue;
            uint32_t end = p;
            while (end < RAM_SIZE >> PAGE_BITS && page_write_stamp[end]) end++;
            fprintf(out, "%s[%" PRIu32 ", %" PRIu32 "]", first ? "" : ", ", p << PAGE_BITS, end << PAGE_BITS);
            first = 0;
        }
        fprintf(out, "]\n}\n");
        fclose(out);
    }
    if (trace_exit)
        fprintf(stderr, "%s %d after %" PRIu64 " instructions, %d frames\n",
                exited ? "exit" : "stopped; no exit", exit_code, total.instructions, frames);
    /* Guest exit codes are reported in the summary; the process status only says
       whether the guest exited (0 for exit 0, 1 for another exit, 3 if stopped). */
    return !exited ? 3 : exit_code != 0;
}
