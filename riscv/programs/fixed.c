/* DOOM's 16.16 fixed-point arithmetic (m_fixed.c): 64-bit products and quotients. */
typedef int fixed_t;
#define FRACBITS 16
#define FRACUNIT (1 << FRACBITS)

static fixed_t FixedMul(fixed_t a, fixed_t b) { return ((long long)a * (long long)b) >> FRACBITS; }

static fixed_t FixedDiv(fixed_t a, fixed_t b) {
    if ((a < 0 ? -a : a) >> 14 >= (b < 0 ? -b : b)) return (a ^ b) < 0 ? -0x7FFFFFFF - 1 : 0x7FFFFFFF;
    return (fixed_t)(((long long)a << FRACBITS) / b);
}

static volatile fixed_t inputs[][2] = {
    {3 * FRACUNIT / 2, 5 * FRACUNIT}, {-7 * FRACUNIT, FRACUNIT / 4}, {123456, -654321},
    {FRACUNIT, 3 * FRACUNIT}, {-FRACUNIT / 3, -FRACUNIT / 7},
};

int main(void) {
    /* Expected values computed by a host build of the same functions. */
    static const fixed_t products[] = {491520, -114688, -1232603, 196608, 3120};
    static const fixed_t quotients[] = {19660, -1835008, -12365, 21845, 152919};
    for (int i = 0; i < 5; i++) {
        if (FixedMul(inputs[i][0], inputs[i][1]) != products[i]) return 1 + i;
        if (FixedDiv(inputs[i][0], inputs[i][1]) != quotients[i]) return 11 + i;
    }
    return 0;
}
