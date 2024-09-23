#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define ELEMENT_SIZE sizeof(double)

double perform_multiplication(size_t array_size, size_t stride) {
    double *A = (double *)malloc(array_size * ELEMENT_SIZE);
    double *B = (double *)malloc(array_size * ELEMENT_SIZE);
    double *C = (double *)malloc(array_size * ELEMENT_SIZE);
    double sum = 0.0;

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays
    for (size_t i = 0; i < array_size; ++i) {
        A[i] = i * 0.5;
        B[i] = (array_size - i) * 0.5;
    }

    // Fixed number of iterations
    size_t iterations = array_size;

    // Multiplication with varying stride
    for (size_t i = 0; i < iterations; ++i) {
        size_t index = (i * stride) % array_size;
        C[index] = A[index] * B[index];
        sum += C[index];
    }

    free(A);
    free(B);
    free(C);

    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <array_size> <stride>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t array_size = atol(argv[1]);
    size_t stride = atol(argv[2]);
    struct timeval start, end;

    // Ensure stride is at least 1
    if (stride == 0) {
        fprintf(stderr, "Stride must be at least 1\n");
        return EXIT_FAILURE;
    }

    gettimeofday(&start, NULL);
    double sum = perform_multiplication(array_size, stride);
    gettimeofday(&end, NULL);

    double elapsed_time = (end.tv_sec - start.tv_sec) +
                          ((end.tv_usec - start.tv_usec) / 1e6);

    printf("Array Size: %zu, Stride: %zu, Time Taken: %f seconds, Sum: %f\n",
           array_size, stride, elapsed_time, sum);

    return EXIT_SUCCESS;
}
