#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <x86intrin.h> 

#define CACHE_LINE_SIZE 64
#define NUM_ITERATIONS 1000000
#define ARRAY_SIZE (64 * 1024 * 1024)  // 64MB

uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void init_array(uint64_t *arr, size_t size, size_t stride) {
    for (size_t i = 0; i < size; i += stride / sizeof(uint64_t)) {
        arr[i] = (i + stride / sizeof(uint64_t)) % size;
    }
}

uint64_t measure_read_latency(uint64_t *arr, size_t size, size_t stride) {
    uint64_t start, end;
    uint64_t index = 0;
    
    start = rdtsc();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        index = arr[index];
    }
    end = rdtsc();
    
    // Prevent optimization
    if (index == (uint64_t)-1) printf("This should never print\n");
    
    return (end - start) / NUM_ITERATIONS;
}

uint64_t measure_write_latency(uint64_t *arr, size_t size, size_t stride) {
    uint64_t start, end;
    uint64_t index = 0;
    uint64_t value = 0;
    
    start = rdtsc();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        arr[index] = value++;
        _mm_mfence();  // Memory fence to ensure write is completed
        index = (index + stride / sizeof(uint64_t)) % size;
    }
    end = rdtsc();
    
    // Prevent optimization
    if (arr[0] == (uint64_t)-1) printf("This should never print\n");
    
    return (end - start) / NUM_ITERATIONS;
}

int main() {
    uint64_t *arr = aligned_alloc(CACHE_LINE_SIZE, ARRAY_SIZE);
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Pin to a single CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    printf("Memory Read and Write Latency Test\n");
    printf("----------------------------------\n");

    size_t strides[] = {64, 512, 4096, 32768};  // in bytes
    const char *names[] = {"L1 Cache", "L2 Cache", "L3 Cache", "Main Memory"};

    for (int i = 0; i < sizeof(strides) / sizeof(strides[0]); i++) {
        size_t stride = strides[i];
        init_array(arr, ARRAY_SIZE / sizeof(uint64_t), stride);

        uint64_t read_latency = measure_read_latency(arr, ARRAY_SIZE / sizeof(uint64_t), stride);
        uint64_t write_latency = measure_write_latency(arr, ARRAY_SIZE / sizeof(uint64_t), stride);

        printf("%s (stride %zu bytes):\n", names[i], stride);
        printf("  Read Latency: %f ns\n", read_latency / 2.112);
        printf("  Write Latency: %f ns\n", write_latency / 2.112);
    }

    free(arr);
    return 0;
}