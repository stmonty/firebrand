/* Memory traffic: insertion sort of a pseudo-random array, then a sortedness check. */
#define N 24

static unsigned values[N];

int main(void) {
    unsigned state = 12345;
    for (int i = 0; i < N; i++) {
        state = state * 1103515245u + 12345u;          /* Linear congruential generator. */
        values[i] = state >> 16;
    }
    for (int i = 1; i < N; i++) {
        unsigned v = values[i];
        int j = i - 1;
        for (; j >= 0 && values[j] > v; j--) values[j + 1] = values[j];
        values[j + 1] = v;
    }
    for (int i = 1; i < N; i++)
        if (values[i - 1] > values[i]) return 1;
    return values[0] == 3498 ? 0 : 2;          /* Minimum, computed by a host build. */
}
