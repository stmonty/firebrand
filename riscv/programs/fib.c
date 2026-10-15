/* Recursion and the stack: naive Fibonacci, checked against an iterative loop. */
static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int main(void) {
    int a = 0, b = 1;
    for (int n = 0; n < 12; n++) {
        if (fib(n) != a) return 1 + n;
        int next = a + b;
        a = b;
        b = next;
    }
    return 0;
}
