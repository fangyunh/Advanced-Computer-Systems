#include <iostream>
#include <chrono>
#include "matrix_multi.hh"

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    ProgramOptions options = parseCommandLineArguments(argc, argv);

    // Generate matrices (no need to measure time)
    Matrix matA = generateMatrix(options.n, options.matAType);
    Matrix matB = generateMatrix(options.n, options.matBType);

    // Placeholder for matrix multiplication
    auto startTime = std::chrono::high_resolution_clock::now();
    Matrix result = multiplyMatrices(matA, matB, options);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> genDuration = endTime - startTime;

    // Calculate total operations
    size_t totalOperations = result.multiplicationCount + result.additionCount;

    // Calculate throughput (operations per second)
    double throughput = totalOperations / genDuration.count();

    // Print settings
    std::cout << "Matrix size: " << options.n << " x " << options.n << "\n";
    std::cout << "Matrix A type: " << (options.matAType == DENSE ? "Dense" : "Sparse") << "\n";
    std::cout << "Matrix B type: " << (options.matBType == DENSE ? "Dense" : "Sparse") << "\n";
    std::cout << "Multithreading: " << (options.enableMultithreading ? "Enabled" : "Disabled") << "\n";
    if (options.enableMultithreading) {
        std::cout << "Number of threads: " << options.threadNum << "\n";
    }
    std::cout << "SIMD: " << (options.enableSIMD ? "Enabled" : "Disabled") << "\n";
    std::cout << "Cache Optimization: " << (options.enableOptimizations ? "Enabled" : "Disabled") << "\n";
    std::cout << "Time Taken: " << genDuration.count() << "s." << std::endl;
    std::cout << "Throughput: " << throughput << " operations/second\n";

    return 0;
}


// g++ -o matrix_mul matrix_main.cc matrix_multi.cc -fopenmp -mavx2 -march=native -std=c++17 -O3 -g