/* cache probe (D115): random pointer chase over a buffer; prints nanoseconds per access.
 * Usage: pchase <bytes> <steps>. The cycle is a single random permutation (Sattolo, fixed seed), so
 * every access depends on the previous one. Built on the target machine only. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t rng = 88172645463325252ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <bytes> <steps>\n", argv[0]);
        return 2;
    }
    size_t n = (size_t)atoll(argv[1]) / sizeof(size_t);
    long long steps = atoll(argv[2]);
    if (n < 2) return 2;
    size_t *a = malloc(n * sizeof(size_t));
    if (!a) return 3;
    for (size_t i = 0; i < n; i++) a[i] = i;
    for (size_t i = n - 1; i > 0; i--) {  /* Sattolo: one cycle through all elements */
        size_t j = (size_t)(next() % i);
        size_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
    struct timespec t0, t1;
    size_t p = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long long k = 0; k < steps; k++) p = a[p];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = 1e9 * (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec);
    printf("end %zu\nns_per_access %.4f\n", p, ns / (double)steps);
    free(a);
    return 0;
}
