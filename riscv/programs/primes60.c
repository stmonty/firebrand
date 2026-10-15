/* Count the primes up to 60, as in lesson 33's @program version. */
volatile int N = 60;
static unsigned char composite[61];
int main(void) {
    int n = N, count = 0;
    for (int p = 2; p <= n; p++) {
        if (composite[p]) continue;
        count++;
        for (int m = p * 2; m <= n; m += p) composite[m] = 1;
    }
    return count;
}
