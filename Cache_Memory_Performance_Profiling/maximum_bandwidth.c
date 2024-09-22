#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE (1UL << 28) // Adjust as needed (e.g., 256MB)

void memory_access(size_t access_size, double read_ratio) {
    char *array = (char *)malloc(ARRAY_SIZE);
    if (!array) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    size_t num_accesses = ARRAY_SIZE / access_size;
    size_t num_reads = num_accesses * read_ratio;
    size_t num_writes = num_accesses - num_reads;

    // Initialize array to prevent optimization
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        array[i] = (char)(i % 256);
    }

    clock_t start = clock();

    // Perform reads
    volatile char temp;
    for (size_t i = 0; i < num_reads; i++) {
        temp = array[(i * access_size) % ARRAY_SIZE];
    }

    // Perform writes
    for (size_t i = 0; i < num_writes; i++) {
        array[(i * access_size) % ARRAY_SIZE] = (char)((i + temp) % 256);
    }

    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    double total_data = (num_reads + num_writes) * access_size / (1024.0 * 1024.0 * 1024.0); // In GB
    double bandwidth = total_data / duration; // In GB/s

    printf("Access Size: %zu Bytes, Read Ratio: %.2f%%\n", access_size, read_ratio * 100);
    printf("Time Taken: %.3f seconds\n", duration);
    printf("Bandwidth: %.3f GB/s\n", bandwidth);

    free(array);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <access_size_in_bytes> <read_ratio>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    size_t access_size = atoi(argv[1]);
    double read_ratio = atof(argv[2]);

    memory_access(access_size, read_ratio);

    return 0;
}
