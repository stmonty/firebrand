/* Byte stores and loads: sieve of Eratosthenes, counting primes below 1000. */
static unsigned char composite[1000];

int main(void) {
    int count = 0;
    for (int i = 2; i < 1000; i++) {
        if (composite[i]) continue;
        count++;
        for (int j = i * i; j < 1000; j += i) composite[j] = 1;
    }
    return count == 168 ? 0 : 1;
}
