#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply_arrays(size_t size) {
    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));
    double *C = (double *)malloc(size * sizeof(double));

    // Initialize arrays
    for (size_t i = 0; i < size; ++i) {
        A[i] = (double)i;
        B[i] = (double)(size - i);
    }

    // Perform multiplication
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t size = atol(argv[1]);
    clock_t start, end;

    start = clock();
    multiply_arrays(size);
    end = clock();

    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Array Size: %zu, Time Taken: %f seconds\n", size, elapsed_time);

    return EXIT_SUCCESS;
}
