/* cpu probe (D115): a fixed amount of integer and floating-point work; prints elapsed seconds.
 * Usage: cpu_work <iterations>. Built on the target machine only (cluster/setup_target.sh). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <iterations>\n", argv[0]);
        return 2;
    }
    long long n = atoll(argv[1]);
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    volatile double x = 1.0;
    unsigned long long h = 1469598103934665603ULL;
    for (long long i = 0; i < n; i++) {
        h = (h ^ (unsigned long long)i) * 1099511628211ULL;
        x = x * 1.0000001 + (double)(h & 7) * 1e-9;
    }
    clock_gettime(CLOCK_MONOTONIC, &b);
    double s = (double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec);
    printf("checksum %llu %.6f\nseconds %.6f\n", h, (double)x, s);
    return 0;
}
