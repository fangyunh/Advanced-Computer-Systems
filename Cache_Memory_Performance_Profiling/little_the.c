// memory_test.c
#define _POSIX_C_SOURCE 199309L

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define DATA_SIZE (512 * 1024 * 1024) // 512 MB
#define TOTAL_OPERATIONS 1000000000UL // 1,000,000,000

int *data_array;
int num_threads;

typedef struct {
    int thread_id;
    unsigned long operations_per_thread;
    double latency_sum;
} thread_arg_t;

void *memory_worker(void *arg) {
    thread_arg_t *t_arg = (thread_arg_t *)arg;
    int thread_id = t_arg->thread_id;
    unsigned long ops_per_thread = t_arg->operations_per_thread;
    unsigned int seed = thread_id;
    volatile int sum = 0;
    struct timespec op_start, op_end;
    double latency_sum = 0.0;

    for (unsigned long i = 0; i < ops_per_thread; i++) {
        int index = rand_r(&seed) % (DATA_SIZE / sizeof(int));

        // Measure latency for a subset of operations
        if (i < 1000) { // Measure the first 1000 operations
            clock_gettime(CLOCK_MONOTONIC, &op_start);
            sum += data_array[index];
            data_array[index] = sum;
            clock_gettime(CLOCK_MONOTONIC, &op_end);
            double op_latency = (op_end.tv_sec - op_start.tv_sec) +
                                (op_end.tv_nsec - op_start.tv_nsec) / 1e9;
            latency_sum += op_latency;
        } else {
            sum += data_array[index];
            data_array[index] = sum;
        }
    }
    t_arg->latency_sum = latency_sum;
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    num_threads = atoi(argv[1]);
    pthread_t threads[num_threads];
    thread_arg_t thread_args[num_threads];

    // Allocate and initialize data array
    data_array = malloc(DATA_SIZE);
    if (!data_array) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < DATA_SIZE / sizeof(int); i++) {
        data_array[i] = i;
    }

    // Calculate operations per thread
    unsigned long operations_per_thread = TOTAL_OPERATIONS / num_threads;

    // Start timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].operations_per_thread = operations_per_thread;
        thread_args[i].latency_sum = 0.0;
        pthread_create(&threads[i], NULL, memory_worker, &thread_args[i]);
    }

    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = (end.tv_sec - start.tv_sec) +
                          (end.tv_nsec - start.tv_nsec) / 1e9;

    // Calculate throughput and latency
    double total_operations = operations_per_thread * num_threads;
    double throughput = total_operations / elapsed_time;

    // Calculate average latency per operation
    double total_latency = 0.0;
    unsigned long measured_operations = 1000 * num_threads;
    for (int i = 0; i < num_threads; i++) {
        total_latency += thread_args[i].latency_sum;
    }
    double average_latency = total_latency / measured_operations;

    // Output results
    printf("Threads: %d\n", num_threads);
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);
    printf("Throughput: %.2f operations/second\n", throughput);
    printf("Average Latency per Operation: %.9f seconds\n", average_latency);

    // Clean up
    free(data_array);
    return 0;
}
