#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>  // Include this header for gettimeofday()

double multiply_arrays(size_t size) {
    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));
    double *C = (double *)malloc(size * sizeof(double));
    double sum = 0.0;

    // Initialize arrays
    for (size_t i = 0; i < size; ++i) {
        A[i] = (double)i;
        B[i] = (double)(size - i);
    }

    // Perform multiplication and accumulate sum
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
        sum += C[i];  // Accumulate the sum
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return sum;  // Return the sum to be used in main
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t size = atol(argv[1]);
    struct timeval start, end;
    double sum;

    gettimeofday(&start, NULL);
    sum = multiply_arrays(size);
    gettimeofday(&end, NULL);

    double elapsed_time = (end.tv_sec - start.tv_sec) +
                          (end.tv_usec - start.tv_usec) / 1e6;
    printf("Array Size: %zu, Time Taken: %f seconds, Sum: %f\n", size, elapsed_time, sum);

    return EXIT_SUCCESS;
}
