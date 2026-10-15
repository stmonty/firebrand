/* GCD by subtraction, as in lesson 33's @program version. */
volatile int A = 60, B = 48;
int main(void) {
    int a = A, b = B;
    while (a != 0 && b != 0) {
        if (a > b) a -= b; else b -= a;
    }
    return a + b;
}
