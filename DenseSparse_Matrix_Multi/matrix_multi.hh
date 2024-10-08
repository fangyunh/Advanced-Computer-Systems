#ifndef MATRIX_MULTI_HH
#define MATRIX_MULTI_HH

#include <vector>
#include <unordered_map>
#include <cstddef>
#include <random>
#include <chrono>
#include <string>
#include <unordered_set>
#include <map>

enum MatrixType {
    DENSE,
    SPARSE
};

// Struct to hold program options
struct ProgramOptions {
    size_t n;                       // Matrix size
    bool enableMultithreading;      // Multithreading flag
    bool enableSIMD;                // SIMD optimization flag
    bool enableOptimizations;            // Cache optimization flag
    size_t threadNum;               // Number of threads
    MatrixType matAType;            // Type of matrix A
    MatrixType matBType;            // Type of matrix B
};

class Matrix {
public:
    MatrixType type;
    size_t n; // Matrix dimensions (n x n)
    bool compressed; // Indicates if the matrix is compressed

    // Dense representation
    std::vector<int> denseData;

    // Sparse representation (Coordinate List)
    std::unordered_map<size_t, int> sparseData;

    // Compressed representation (CSR format)
    std::vector<int> values;
    std::vector<size_t> colIdx;
    std::vector<size_t> rowPtr;

    // Constructor
    Matrix(size_t size, MatrixType matrixType, bool compressed = false);

    // Methods
    void setElement(size_t row, size_t col, int value);
    int getElement(size_t row, size_t col) const;
    void compress(); // Compress the matrix
};

// Function to generate a matrix with specified size and type
Matrix generateMatrix(size_t n, MatrixType matrixType);

// Functions for different options
Matrix multiply(const Matrix& matA, const Matrix& matB);
Matrix multiplyM(const Matrix& matA, const Matrix& matB, size_t threadNum);
Matrix multiplyS(const Matrix& matA, const Matrix& matB);
Matrix multiplyO(Matrix& matA, Matrix& matB);
Matrix multiplyMS(const Matrix& matA, const Matrix& matB, size_t threadNum);
Matrix multiplyMO(Matrix& matA, Matrix& matB, size_t threadNum);
Matrix multiplySO(Matrix& matA, Matrix& matB);
Matrix multiplyMSO(Matrix& matA, Matrix& matB, size_t threadNum);

// Main multiplication function
Matrix multiplyMatrices(Matrix& matA, Matrix& matB, const ProgramOptions& options);

// Process input
ProgramOptions parseCommandLineArguments(int argc, char* argv[]);

#endif // MATRIX_MULTI_HH