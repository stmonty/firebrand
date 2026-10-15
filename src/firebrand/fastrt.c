/*
 * Compiled decoder for firebrand models (backend="c" in Transformer.generate).
 *
 * Runs the same weights and schedule as runtime.py, one position at a time.
 * Heads find their winner with an exact index where the key geometry allows
 * (absolute reads, latest-value arrays, match hash tables) and scan otherwise.
 * Evaluation is incremental: a unit whose inputs did not change since the
 * previous position is skipped, and units that read one small-range row are
 * memoized. All arithmetic is float64 on integers below 2^53, so results do not
 * depend on evaluation order.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t n; const int32_t *row, *col; const double *val; } Coo;

enum { HEAD_ABSOLUTE = 0, HEAD_LATEST = 1, HEAD_SCAN = 2, HEAD_MATCH = 3 };
enum { FEATURE_MOD = 0, FEATURE_TAG = 1, FEATURE_SQUARE = 2, FEATURE_POINT = 3, FEATURE_DIGIT = 4,
       FEATURE_QUOTIENT = 5 };
enum { ERROR_ADDRESS = -1, ERROR_TAG = -2, ERROR_CONTEXT = -3, ERROR_MEMORY = -4, ERROR_SENTINEL = -5 };

typedef struct {
    int32_t width, input_width, vocab, lower, max_context;
    int32_t embedding_columns;
    const double *embedding;                    /* vocab x embedding_columns */
    int32_t features;
    const int32_t *feature;                     /* features x 6: kind, column, a, b, c, d */
    Coo unembedding;
    int32_t heads;
    const int32_t *head_kind, *head_dims;
    const int32_t *match;                       /* heads x 6: digits, period, phase, start, sentinel, end */
    const double *match_scale;                  /* per head: M in the match score */
    const Coo *q, *k, *v, *o;
    int32_t halts;
    const int32_t *halt;
    /* Units in schedule order. A head unit names its head; an FFN unit owns
       hidden neurons with input entries (row, neuron, weight) and output
       entries (neuron, slot, weight), neurons numbered from 0 per unit. */
    int32_t units;
    const int32_t *unit_head;                   /* head index, or -1 for an FFN unit */
    const int32_t *unit_hidden;
    const int32_t *in_start, *in_row, *in_neuron;
    const double *in_value;
    const int32_t *out_start, *out_neuron, *out_slot;
    const double *out_value;
    /* Per unit: the distinct slots it writes. Per residual row: the units that read it. */
    const int32_t *write_start, *writes;
    const int32_t *reader_start, *readers;
    /* Per unit: the one row a memoized FFN unit reads besides the constant (or
       -1), that row's lowest value, and the unit's table offsets (units + 1). */
    const int32_t *memo_row, *memo_low, *memo_start;
    /* Likewise the output token, when the unembedding reads one such row. */
    int32_t output_row, output_low, output_span;
} Model;

/* Input coordinates of one position: embedding row plus positional features. */
static void input_vector(const Model *m, int32_t token, int64_t position, double *x) {
    const double *row = m->embedding + (int64_t)(token - m->lower) * m->embedding_columns;
    int32_t n = m->embedding_columns < m->input_width ? m->embedding_columns : m->input_width;
    memcpy(x, row, sizeof(double) * n);
    for (int32_t i = n; i < m->input_width; i++) x[i] = 0;
    x[1] = (double)position;
    x[2] = (double)position * (double)position;
    for (int32_t f = 0; f < m->features; f++) {
        const int32_t *d = m->feature + 6 * f;
        int64_t shifted = position + d[3];
        if (d[0] == FEATURE_MOD) x[d[1]] = (double)((position + d[3]) % d[2]);
        else if (d[0] == FEATURE_TAG)
            x[d[1]] = position >= d[4] && (d[5] < 0 || position < d[5]) && position % d[2] == d[3];
        else if (d[0] == FEATURE_POINT) x[d[1]] = position == d[2];
        else if (d[0] == FEATURE_DIGIT) x[d[1]] = shifted >= 0 ? (double)(shifted >> (8 * d[2]) & 255) : 0;
        else if (d[0] == FEATURE_QUOTIENT) x[d[1]] = shifted >= 0 ? (double)(shifted / d[2]) : -1;
    }
}

static void coo_apply(const Coo *c, const double *x, double *y) {
    for (int32_t i = 0; i < c->n; i++) y[c->col[i]] += x[c->row[i]] * c->val[i];
}

typedef struct {
    int32_t *newest;          /* Newest tagged position for each token value, or -1. */
    int32_t any;
} LatestIndex;

/* Open-addressing map from packed key digits to the newest matching position. */
typedef struct {
    uint64_t *keys;
    int32_t *positions;
    int64_t capacity, used;
    int64_t sentinel;
} MatchIndex;

static uint64_t mix(uint64_t key) {
    key ^= key >> 33; key *= 0xff51afd7ed558ccdull; key ^= key >> 33;
    return key;
}

static int match_put(MatchIndex *index, uint64_t key, int32_t position) {
    if (2 * (index->used + 1) > index->capacity) {
        int64_t capacity = index->capacity ? 2 * index->capacity : 1024;
        uint64_t *keys = malloc(sizeof(uint64_t) * capacity);
        int32_t *positions = malloc(sizeof(int32_t) * capacity);
        if (!keys || !positions) { free(keys); free(positions); return ERROR_MEMORY; }
        for (int64_t i = 0; i < capacity; i++) positions[i] = -1;
        for (int64_t i = 0; i < index->capacity; i++) {
            if (index->positions[i] < 0) continue;
            int64_t slot = mix(index->keys[i]) & (capacity - 1);
            while (positions[slot] >= 0) slot = (slot + 1) & (capacity - 1);
            keys[slot] = index->keys[i];
            positions[slot] = index->positions[i];
        }
        free(index->keys); free(index->positions);
        index->keys = keys; index->positions = positions; index->capacity = capacity;
    }
    int64_t slot = mix(key) & (index->capacity - 1);
    while (index->positions[slot] >= 0 && index->keys[slot] != key) slot = (slot + 1) & (index->capacity - 1);
    if (index->positions[slot] < 0) index->used++;
    index->keys[slot] = key;
    index->positions[slot] = position;
    return 0;
}

static int64_t match_get(const MatchIndex *index, uint64_t key) {
    if (index->capacity) {
        int64_t slot = mix(key) & (index->capacity - 1);
        while (index->positions[slot] >= 0) {
            if (index->keys[slot] == key) return index->positions[slot];
            slot = (slot + 1) & (index->capacity - 1);
        }
    }
    return index->sentinel >= 0 ? index->sentinel : ERROR_SENTINEL;
}

static uint64_t pack_digits(const double *digits, int32_t count, double divisor) {
    uint64_t key = 0;
    for (int32_t i = 0; i < count; i++) key |= (uint64_t)(int64_t)(digits[i] / divisor) << (16 * i);
    return key;
}

static void project(const Model *m, const Coo *c, int32_t dims, const int32_t *tokens, int64_t p,
                    double *scratch, double *out) {
    input_vector(m, tokens[p], p, scratch);
    for (int32_t d = 0; d < dims; d++) out[d] = 0;
    coo_apply(c, scratch, out);
}

static void index_add(const Model *m, LatestIndex *index, int h, const int32_t *tokens, int64_t p,
                      double *scratch, int64_t *version) {
    double key[8];
    project(m, &m->k[h], m->head_dims[h], tokens, p, scratch, key);
    if (key[3] == 1.0) {
        int64_t value = (int64_t)key[0] - m->lower;
        if (value >= 0 && value < m->vocab) {
            index->newest[value] = (int32_t)key[2];
            index->any = 1;
            (*version)++;
        }
    }
}

/* Winner of a latest head: nearest value to key, then the newest position. */
static int64_t index_query(const Model *m, const LatestIndex *index, double key_value) {
    if (!index->any) return ERROR_TAG;
    int64_t key = (int64_t)key_value - m->lower;
    if (key >= 0 && key < m->vocab && index->newest[key] >= 0) return index->newest[key];
    for (int64_t distance = 1;; distance++) {
        int64_t below = key - distance, above = key + distance;
        int32_t a = below >= 0 && below < m->vocab ? index->newest[below] : -1;
        int32_t b = above >= 0 && above < m->vocab ? index->newest[above] : -1;
        if (a >= 0 || b >= 0) return a > b ? a : b;
        if (below < 0 && above >= m->vocab) return ERROR_TAG;
    }
}

typedef struct {
    LatestIndex *latest;
    MatchIndex *match;
    double *x, *scratch, *logits, *hidden, *saved, *input;
    double *memo;                     /* Memoized units' outputs by input value; NaN: not yet seen. */
    int32_t *output_memo;             /* Output token index by output row value; -1: not yet seen. */
    uint64_t *dirty;                  /* Bit per unit: an input changed during this pass. */
    int32_t dirty_words;
    int32_t *head_unit;               /* The unit of each head (or -1). */
    int64_t *version, *seen;          /* Index versions per head; version each head last used. */
    int fresh;                        /* The next pass must compute everything. */
    int64_t units_run, units_skipped, memo_hits;
} State;

/* A residual row changed: every unit that reads it must run. Readers always
   come later in schedule order, so a forward scan of the bitset finds them. */
static void mark_readers(const Model *m, State *st, int32_t row) {
    for (int32_t i = m->reader_start[row]; i < m->reader_start[row + 1]; i++) {
        int32_t u = m->readers[i];
        st->dirty[u >> 6] |= 1ull << (u & 63);
    }
}

static void mark_unit(State *st, int32_t u) { st->dirty[u >> 6] |= 1ull << (u & 63); }

/* One attention head: find its winner and write its value into its slot. */
static int run_head(const Model *m, State *st, int32_t h, const int32_t *tokens, int64_t length) {
    double *x = st->x, q[16] = {0}, value[8];
    int32_t dims = m->head_dims[h];
    int64_t winner = -1;
    coo_apply(&m->q[h], x, q);
    if (m->head_kind[h] == HEAD_ABSOLUTE) {
        double address = q[0];
        if (!isfinite(address) || address != floor(address) || address < 0 || address >= length)
            return ERROR_ADDRESS;
        winner = (int64_t)address;
    } else if (m->head_kind[h] == HEAD_MATCH) {
        const int32_t *meta = m->match + 6 * h;
        int32_t digits = meta[0];
        double key[16] = {0};
        MatchIndex *index = &st->match[h];
        coo_apply(&m->k[h], x, key);                  /* This position's own key. */
        if (key[2 * digits + 2] == 1.0) {
            if (match_put(index, pack_digits(key, digits, 1.0), (int32_t)(length - 1))) return ERROR_MEMORY;
            st->version[h]++;
        }
        if (key[2 * digits] == 1.0) { index->sentinel = length - 1; st->version[h]++; }
        winner = match_get(index, pack_digits(q, digits, 2 * m->match_scale[h]));
        if (winner < 0) return (int)winner;
    } else if (m->head_kind[h] == HEAD_LATEST && q[0] / (-2 * q[1]) == floor(q[0] / (-2 * q[1]))) {
        winner = index_query(m, &st->latest[h], q[0] / (-2 * q[1]));
        if (winner < 0) return (int)winner;
    } else {
        double best = -INFINITY;
        for (int64_t p = 0; p < length; p++) {
            double key[8], score = 0;
            project(m, &m->k[h], dims, tokens, p, st->scratch, key);
            for (int32_t d = 0; d < dims; d++) score += q[d] * key[d];
            if (score > best) { best = score; winner = p; }
        }
        if (m->head_kind[h] == HEAD_LATEST) {
            double key[8];
            project(m, &m->k[h], dims, tokens, winner, st->scratch, key);
            if (key[3] != 1.0) return ERROR_TAG;
        }
    }
    project(m, &m->v[h], 1, tokens, winner, st->scratch, value);
    coo_apply(&m->o[h], value, x);
    st->seen[h] = st->version[h];
    return 0;
}

/* One FFN node's neurons. */
static void run_ffn(const Model *m, State *st, int32_t u) {
    double *x = st->x, *hidden = st->hidden;
    /* A memoized unit computes a function of one integer row: look it up, or
       compute it once and remember it. Values outside the table are computed. */
    double *cell = NULL;
    if (m->memo_row[u] >= 0) {
        double v = x[m->memo_row[u]] - m->memo_low[u];
        if (v >= 0 && v < m->memo_start[u + 1] - m->memo_start[u] && v == (double)(int32_t)v) {
            cell = st->memo + m->memo_start[u] + (int32_t)v;
            if (!isnan(*cell)) {
                x[m->writes[m->write_start[u]]] = *cell;
                st->memo_hits++;
                return;
            }
        }
    }
    memset(hidden, 0, sizeof(double) * m->unit_hidden[u]);
    for (int32_t i = m->in_start[u]; i < m->in_start[u + 1]; i++)
        hidden[m->in_neuron[i]] += x[m->in_row[i]] * m->in_value[i];
    for (int32_t i = m->out_start[u]; i < m->out_start[u + 1]; i++) {
        double h = hidden[m->out_neuron[i]];
        if (h > 0) x[m->out_slot[i]] += h * m->out_value[i];
    }
    if (cell) *cell = x[m->writes[m->write_start[u]]];
}

/*
 * One forward pass at position length - 1. Registers that position's match keys
 * and stores the argmax token; returns 0 or a negative error code. (Tokens can
 * be negative, so they never share the return value with errors.)
 */
static int forward(const Model *m, State *st, const int32_t *tokens, int64_t length, int32_t *token) {
    double *x = st->x;
    /* New input coordinates; changes mark the units that read them. */
    input_vector(m, tokens[length - 1], length - 1, st->input);
    memset(st->dirty, 0, sizeof(uint64_t) * st->dirty_words);
    for (int32_t i = 0; i < m->input_width; i++) {
        if (st->input[i] != x[i]) mark_readers(m, st, i);
        x[i] = st->input[i];
    }
    if (st->fresh) {
        for (int32_t u = 0; u < m->units; u++) mark_unit(st, u);
    } else {
        for (int32_t h = 0; h < m->heads; h++)
            if (st->head_unit[h] >= 0 && (m->head_kind[h] == HEAD_SCAN || st->version[h] != st->seen[h]))
                mark_unit(st, st->head_unit[h]);
    }
    int64_t run_before = st->units_run;
    for (int32_t word = 0; word < st->dirty_words; word++)
    while (st->dirty[word]) {
        int32_t u = word * 64 + __builtin_ctzll(st->dirty[word]);
        st->dirty[word] &= st->dirty[word] - 1;
        int32_t h = m->unit_head[u];
        st->units_run++;
        int32_t first = m->write_start[u], count = m->write_start[u + 1] - first;
        for (int32_t i = 0; i < count; i++) {
            st->saved[i] = x[m->writes[first + i]];
            x[m->writes[first + i]] = 0;
        }
        if (h >= 0) {
            int r = run_head(m, st, h, tokens, length);
            if (r < 0) return r;
        } else {
            run_ffn(m, st, u);
        }
        for (int32_t i = 0; i < count; i++)
            if (x[m->writes[first + i]] != st->saved[i]) mark_readers(m, st, m->writes[first + i]);
    }
    st->units_skipped += m->units - (st->units_run - run_before);
    st->fresh = 0;
    int32_t *cell = NULL;
    if (m->output_row >= 0) {
        double v = x[m->output_row] - m->output_low;
        if (v >= 0 && v < m->output_span && v == (double)(int32_t)v) {
            cell = st->output_memo + (int32_t)v;
            if (*cell >= 0) { *token = *cell + m->lower; return 0; }
        }
    }
    memset(st->logits, 0, sizeof(double) * m->vocab);
    coo_apply(&m->unembedding, x, st->logits);
    int32_t best = 0;
    for (int32_t i = 1; i < m->vocab; i++) if (st->logits[i] > st->logits[best]) best = i;
    *token = best + m->lower;
    if (cell) *cell = best;
    return 0;
}

static int tagged_for_match(const Model *m, int64_t p) {
    for (int32_t h = 0; h < m->heads; h++) {
        if (m->head_kind[h] != HEAD_MATCH) continue;
        const int32_t *meta = m->match + 6 * h;
        if (p >= meta[3] && (meta[5] < 0 || p < meta[5]) && p % meta[1] == meta[2]) return 1;
    }
    return 0;
}

/* Statistics from the last call: units evaluated and units skipped. */
int64_t fastrt_units_run, fastrt_units_skipped, fastrt_memo_hits;

/*
 * Extend tokens[0:length] by up to steps tokens (stopping after a halt token).
 * Returns the new length, or a negative error code.
 */
int64_t fastrt_generate(const Model *m, int32_t *tokens, int64_t length, int64_t steps) {
    State st = {0};
    int32_t hidden_max = 1, writes_max = 1, has_match = 0;
    for (int32_t u = 0; u < m->units; u++) {
        if (m->unit_hidden[u] > hidden_max) hidden_max = m->unit_hidden[u];
        if (m->write_start[u + 1] - m->write_start[u] > writes_max) writes_max = m->write_start[u + 1] - m->write_start[u];
    }
    st.x = calloc(m->width, sizeof(double));
    st.scratch = calloc(m->input_width, sizeof(double));
    st.logits = calloc(m->vocab, sizeof(double));
    st.hidden = calloc(hidden_max, sizeof(double));
    st.saved = calloc(writes_max, sizeof(double));
    st.input = calloc(m->input_width, sizeof(double));
    st.memo = malloc(sizeof(double) * (m->memo_start[m->units] > 0 ? m->memo_start[m->units] : 1));
    for (int32_t i = 0; st.memo && i < m->memo_start[m->units]; i++) st.memo[i] = NAN;
    st.output_memo = malloc(sizeof(int32_t) * (m->output_span > 0 ? m->output_span : 1));
    for (int32_t i = 0; st.output_memo && i < m->output_span; i++) st.output_memo[i] = -1;
    st.dirty_words = (m->units + 63) / 64;
    st.dirty = calloc(st.dirty_words > 0 ? st.dirty_words : 1, sizeof(uint64_t));
    st.latest = calloc(m->heads > 0 ? m->heads : 1, sizeof(LatestIndex));
    st.match = calloc(m->heads > 0 ? m->heads : 1, sizeof(MatchIndex));
    st.version = calloc(m->heads > 0 ? m->heads : 1, sizeof(int64_t));
    st.seen = calloc(m->heads > 0 ? m->heads : 1, sizeof(int64_t));
    st.head_unit = malloc(sizeof(int32_t) * (m->heads > 0 ? m->heads : 1));
    for (int32_t h = 0; st.head_unit && h < m->heads; h++) st.head_unit[h] = -1;
    for (int32_t u = 0; st.head_unit && u < m->units; u++)
        if (m->unit_head[u] >= 0) st.head_unit[m->unit_head[u]] = u;
    st.fresh = 1;
    int64_t result = length;
    if (!st.x || !st.scratch || !st.logits || !st.hidden || !st.saved || !st.input || !st.dirty
        || !st.latest || !st.match || !st.version || !st.seen || !st.head_unit || !st.memo
        || !st.output_memo) {
        result = ERROR_MEMORY; goto done;
    }
    for (int32_t h = 0; h < m->heads; h++) {
        st.match[h].sentinel = -1;
        has_match |= m->head_kind[h] == HEAD_MATCH;
        if (m->head_kind[h] != HEAD_LATEST) continue;
        st.latest[h].newest = malloc(sizeof(int32_t) * m->vocab);
        if (!st.latest[h].newest) { result = ERROR_MEMORY; goto done; }
        for (int32_t i = 0; i < m->vocab; i++) st.latest[h].newest[i] = -1;
    }
    /* Build prompt indexes in position order; a prompt position tagged for a match
       head gets its forward pass (with the indexes as they stood there). */
    for (int64_t p = 0; p < length; p++) {
        for (int32_t h = 0; h < m->heads; h++)
            if (m->head_kind[h] == HEAD_LATEST) index_add(m, &st.latest[h], h, tokens, p, st.scratch, &st.version[h]);
        if (has_match && p < length - 1 && tagged_for_match(m, p)) {
            int32_t ignored;
            int r = forward(m, &st, tokens, p + 1, &ignored);
            if (r < 0) { result = r; goto done; }
        }
    }

    for (int64_t step = 0; step < steps; step++) {
        int32_t last = tokens[length - 1];
        int halted = 0;
        for (int32_t i = 0; i < m->halts; i++) halted |= last == m->halt[i];
        if (halted) break;
        if (length > m->max_context) { result = ERROR_CONTEXT; goto done; }
        int32_t token;
        int r = forward(m, &st, tokens, length, &token);
        if (r < 0) { result = r; goto done; }
        tokens[length] = token;
        for (int32_t h = 0; h < m->heads; h++)
            if (m->head_kind[h] == HEAD_LATEST) index_add(m, &st.latest[h], h, tokens, length, st.scratch, &st.version[h]);
        length++;
        result = length;
    }
done:
    for (int32_t h = 0; st.latest && h < m->heads; h++) free(st.latest[h].newest);
    for (int32_t h = 0; st.match && h < m->heads; h++) { free(st.match[h].keys); free(st.match[h].positions); }
    free(st.latest); free(st.match); free(st.x); free(st.scratch); free(st.logits); free(st.hidden);
    free(st.saved); free(st.input); free(st.memo); free(st.output_memo); free(st.dirty); free(st.version); free(st.seen); free(st.head_unit);
    fastrt_units_run = st.units_run;
    fastrt_units_skipped = st.units_skipped;
    fastrt_memo_hits = st.memo_hits;
    return result;
}
