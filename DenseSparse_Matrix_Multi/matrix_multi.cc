
#include "matrix_multi.hh"
#include <iostream>
#include <string>
#include <unordered_set>
#include <random>
#include <chrono>
#include <cstdlib>
#include <algorithm> 
#include <omp.h>
#include <immintrin.h>
#include <map>

// Function to parse command-line arguments
ProgramOptions 
parseCommandLineArguments(int argc, char* argv[]) 
{
    ProgramOptions options;

    // Initialize default values
    options.n = 0; // Set to 0 to detect if it's set
    options.enableMultithreading = false;
    options.enableSIMD = false;
    options.enableOptimizations = false;
    options.threadNum = 1;
    options.matAType = DENSE;
    options.matBType = DENSE;
    bool nProvided = false;
    bool tProvided = false;

    // Allowed options for -t flag
    std::unordered_set<std::string> validMatrixTypes = {"dd", "ds", "ss"};

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            options.n = static_cast<size_t>(std::stoul(argv[++i]));
            nProvided = true;
        } else if (arg == "-m" && i + 1 < argc) {
            options.enableMultithreading = true;
            options.threadNum = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "-s") {
            options.enableSIMD = true;
        } else if (arg == "-o") {
            options.enableOptimizations = true;
        } else if (arg == "-t" && i + 1 < argc) {
            std::string type = argv[++i];
            if (validMatrixTypes.find(type) != validMatrixTypes.end()) {
                tProvided = true;
                if (type == "dd") {
                    options.matAType = DENSE;
                    options.matBType = DENSE;
                } else if (type == "ds") {
                    options.matAType = DENSE;
                    options.matBType = SPARSE;
                } else if (type == "ss") {
                    options.matAType = SPARSE;
                    options.matBType = SPARSE;
                }
            } else {
                std::cerr << "Invalid matrix type. Use 'dd', 'ds', or 'ss'.\n";
                exit(1);
            }
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            exit(1);
        }
    }

    // Check if -n and -t flags were provided
    if (!nProvided || !tProvided) {
        std::cerr << "Error: Missing required arguments.\n";
        std::cerr << "Usage: ./matrix_mul -n [size] -t [dd|ds|ss] [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  -m [threadNum]  Enable multithreading with specified number of threads.\n";
        std::cerr << "  -s              Enable SIMD optimizations.\n";
        std::cerr << "  -o              Enable data access pattern and compression optimizations.\n";
        exit(1);
    }

    return options;
}

// Constructor for the Matrix class
Matrix::Matrix(size_t size, MatrixType matrixType)
    : n(size), type(matrixType), maxNNZPerRow(0) {
    // Initialize random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Generate matrix data
    generateRandomData();
}

void 
Matrix::setElement(size_t row, size_t col, int value) 
{
    if (compressed) {
        std::cerr << "Error: Cannot set elements in a compressed matrix.\n";
        return;
    }

    // Use ELLPACK format
    size_t offset = row * maxNNZPerRow;
    bool found = false;

    for (size_t k = 0; k < maxNNZPerRow; ++k) {
        if (colIndices[offset + k] == col) {
            values[offset + k] = value;
            found = true;
            break;
        }
    }

    if (!found && value != 0) {
        // Add new non-zero element if there's space
        for (size_t k = 0; k < maxNNZPerRow; ++k) {
            if (values[offset + k] == 0) {
                values[offset + k] = value;
                colIndices[offset + k] = col;
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Error: Exceeded maximum number of non-zero elements per row.\n";
            // Optionally, increase maxNNZPerRow and resize vectors
        }
    }
}



// Get an element from the matrix
int 
Matrix::getElement(size_t row, size_t col) const {
    if (compressed) {
        size_t rowStart = csrRowPtr[row];
        size_t rowEnd = csrRowPtr[row + 1];

        for (size_t idx = rowStart; idx < rowEnd; ++idx) {
            if (csrColIdx[idx] == col) {
                return csrValues[idx];
            }
        }
        return 0; // Element not found, it's zero
    } else {
        // Use ELLPACK format
        size_t offset = row * maxNNZPerRow;
        for (size_t k = 0; k < maxNNZPerRow; ++k) {
            if (colIndices[offset + k] == col) {
                return values[offset + k];
            }
        }
        return 0; // Element not found, it's zero
    }
}



// Function to generate a matrix with specified size and type
void 
Matrix::generateRandomData() 
{
    size_t maxNNZ = 0;

    // Temporary storage for row-wise data
    std::vector<std::vector<std::pair<size_t, int>>> rows(n);

    for (size_t i = 0; i < n; ++i) {
        size_t nnzInRow = 0;

        for (size_t j = 0; j < n; ++j) {
            int value = 0;

            if (type == DENSE) {
                // For dense matrices, fill with random values
                value = std::rand() % 10 + 1; // Random values between 1 and 10
            } else {
                // For sparse matrices, randomly decide if the element is non-zero
                if (std::rand() % 10 == 0) { // 10% chance of being non-zero
                    value = std::rand() % 10 + 1;
                }
            }

            if (value != 0) {
                rows[i].emplace_back(j, value);
                ++nnzInRow;
            }
        }

        if (nnzInRow > maxNNZ) {
            maxNNZ = nnzInRow;
        }
    }

    maxNNZPerRow = maxNNZ;

    // Resize the values and colIndices arrays
    values.resize(n * maxNNZPerRow, 0);
    colIndices.resize(n * maxNNZPerRow, 0);

    // Fill the ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        size_t offset = i * maxNNZPerRow;
        size_t k = 0;

        for (const auto& elem : rows[i]) {
            values[offset + k] = elem.second;
            colIndices[offset + k] = elem.first;
            ++k;
        }

    }
}

Matrix 
generateMatrix(size_t n, MatrixType matrixType) 
{
    Matrix mat(n, matrixType);
    mat.generateRandomData();
    return mat;
}


void 
Matrix::compress() 
{
    if (compressed) return; // Already compressed

    // Clear any existing compressed data
    csrValues.clear();
    csrColIdx.clear();
    csrRowPtr.clear();

    csrRowPtr.reserve(n + 1);
    csrRowPtr.push_back(0); // Start with zero

    for (size_t i = 0; i < n; ++i) {
        size_t nnzInRow = 0;

        size_t offset = i * maxNNZPerRow;
        for (size_t k = 0; k < maxNNZPerRow; ++k) {
            int val = values[offset + k];
            size_t col = colIndices[offset + k];

            if (val != 0) {
                csrValues.push_back(val);
                csrColIdx.push_back(col);
                nnzInRow++;
            }
        }

        csrRowPtr.push_back(csrRowPtr.back() + nnzInRow);
    }

    // Optionally, clear ELLPACK data to save memory
    values.clear();
    values.shrink_to_fit();
    colIndices.clear();
    colIndices.shrink_to_fit();

    compressed = true;
}


// Multiply matrices with consideration of optimizations
Matrix 
multiplyMatrices(Matrix& matA, Matrix& matB, const ProgramOptions& options) 
{
    bool enableMultithreading = options.enableMultithreading;
    bool enableSIMD = options.enableSIMD;
    bool enableOptimizations = options.enableOptimizations;
    size_t threadNum = options.threadNum;

    if (!enableMultithreading && !enableSIMD && !enableOptimizations) {
        // No optimizations
        return multiply(matA, matB);
    } else if (enableMultithreading && !enableSIMD && !enableOptimizations) {
        // Multithreading only
        return multiplyM(matA, matB, threadNum);
    } else if (!enableMultithreading && enableSIMD && !enableOptimizations) {
        // SIMD only
        return multiplyS(matA, matB);
    } else if (!enableMultithreading && !enableSIMD && enableOptimizations) {
        // Optimizations only
        return multiplyO(matA, matB);
    } else if (enableMultithreading && enableSIMD && !enableOptimizations) {
        // Multithreading and SIMD
        return multiplyMS(matA, matB, threadNum);
    } else if (enableMultithreading && !enableSIMD && enableOptimizations) {
        // Multithreading and Optimizations
        return multiplyMO(matA, matB, threadNum);
    } else if (!enableMultithreading && enableSIMD && enableOptimizations) {
        // SIMD and Optimizations
        return multiplySO(matA, matB);
    } else if (enableMultithreading && enableSIMD && enableOptimizations) {
        // Multithreading, SIMD, and Optimizations
        return multiplyMSO(matA, matB, threadNum);
    } else {
        std::cerr << "Invalid combination of options.\n";
        exit(1);
    }
}

Matrix 
multiply(const Matrix& matA, const Matrix& matB) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE); // Result will be in ELLPACK format

    // Initialize result matrix
    result.maxNNZPerRow = n; // Maximum possible non-zero elements per row
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    for (size_t i = 0; i < n; ++i) {
        size_t resultOffset = i * result.maxNNZPerRow;
        size_t resultNNZ = 0;

        for (size_t j = 0; j < n; ++j) {
            int sum = 0;

            for (size_t k = 0; k < n; ++k) {
                int valA = matA.getElement(i, k);
                int valB = matB.getElement(k, j);
                sum += valA * valB;
            }

            if (sum != 0) {
                if (resultNNZ < result.maxNNZPerRow) {
                    result.values[resultOffset + resultNNZ] = sum;
                    result.colIndices[resultOffset + resultNNZ] = j;
                    resultNNZ++;
                } else {
                    // Exceeded maxNNZPerRow, need to handle resizing or report an error
                    std::cerr << "Error: Exceeded maximum non-zero elements per row in result.\n";
                    exit(1);
                }
            }
        }
    }

    return result;
}


Matrix 
multiplyM(const Matrix& matA, const Matrix& matB, size_t threadNum) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Initialize result matrix
    result.maxNNZPerRow = n; // Maximum possible non-zero elements per row
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        size_t resultOffset = i * result.maxNNZPerRow;
        size_t resultNNZ = 0;

        for (size_t j = 0; j < n; ++j) {
            int sum = 0;

            for (size_t k = 0; k < n; ++k) {
                int valA = matA.getElement(i, k);
                int valB = matB.getElement(k, j);
                sum += valA * valB;
            }

            if (sum != 0) {
                result.values[resultOffset + resultNNZ] = sum;
                result.colIndices[resultOffset + resultNNZ] = j;
                resultNNZ++;
            }
        }

        // Update result.maxNNZPerRow if necessary
        // For simplicity, we assume maxNNZPerRow is sufficient
    }
    return result;
}


Matrix 
multiplyS(const Matrix& matA, const Matrix& matB) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            int valA = matA.getElement(i, k);
            __m256i vecA = _mm256_set1_epi32(valA);

            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.denseData[k * n + j]);
                __m256i vecC = _mm256_loadu_si256((__m256i*)&result.denseData[i * n + j]);
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);
                vecC = _mm256_add_epi32(vecC, vecMul);
                _mm256_storeu_si256((__m256i*)&result.denseData[i * n + j], vecC);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                int valB = matB.getElement(k, j);
                result.denseData[i * n + j] += valA * valB;
            }
        }
    }

    return result;
}

Matrix 
multiplyO(Matrix& matA, Matrix& matB) 
{
    // Compress matrices
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.rowPtr[i];
        size_t rowEndA = matA.rowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            size_t k = matA.colIdx[idxA];
            int valA = matA.values[idxA];

            size_t rowStartB = matB.rowPtr[k];
            size_t rowEndB = matB.rowPtr[k + 1];

            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                size_t j = matB.colIdx[idxB];
                int valB = matB.values[idxB];

                result.denseData[i * n + j] += valA * valB;
            }
        }
    }

    return result;
}

Matrix 
multiplyMS(const Matrix& matA, const Matrix& matB, size_t threadNum) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            int valA = matA.getElement(i, k);
            __m256i vecA = _mm256_set1_epi32(valA);

            size_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.denseData[k * n + j]);
                __m256i vecC = _mm256_loadu_si256((__m256i*)&result.denseData[i * n + j]);
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);
                vecC = _mm256_add_epi32(vecC, vecMul);
                _mm256_storeu_si256((__m256i*)&result.denseData[i * n + j], vecC);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                int valB = matB.getElement(k, j);
                result.denseData[i * n + j] += valA * valB;
            }
        }
    }

    return result;
}

Matrix 
multiplyMO(Matrix& matA, Matrix& matB, size_t threadNum) 
{
    // Compress matrices
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.rowPtr[i];
        size_t rowEndA = matA.rowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            size_t k = matA.colIdx[idxA];
            int valA = matA.values[idxA];

            size_t rowStartB = matB.rowPtr[k];
            size_t rowEndB = matB.rowPtr[k + 1];

            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                size_t j = matB.colIdx[idxB];
                int valB = matB.values[idxB];

                #pragma omp atomic
                result.denseData[i * n + j] += valA * valB;
            }
        }
    }

    return result;
}

Matrix 
multiplySO(Matrix& matA, Matrix& matB) 
{
    // Compress matrices if not already compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Allocate memory without alignment
    result.denseData.resize(n * n, 0);

    // Proceed with SIMD code using unaligned load/store
    const size_t SIMD_WIDTH = 8; // AVX2 processes 8 integers

    // Loop over rows of matA
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.rowPtr[i];
        size_t rowEndA = matA.rowPtr[i + 1];

        // Loop over non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            size_t k = matA.colIdx[idxA];  // Column index in matA
            int valA = matA.values[idxA];  // Non-zero value in matA

            // Get the row k of matB
            size_t rowStartB = matB.rowPtr[k];
            size_t rowEndB = matB.rowPtr[k + 1];

            size_t idxB = rowStartB;

            // Process elements in chunks of SIMD_WIDTH
            while (idxB + SIMD_WIDTH <= rowEndB) {
                // Load column indices and values from matB
                __m256i vecIndices = _mm256_loadu_si256((__m256i*)&matB.colIdx[idxB]);
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.values[idxB]);

                // Multiply valA with vecB
                __m256i vecA = _mm256_set1_epi32(valA);
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);

                // Compute result indices
                __m256i rowOffset = _mm256_set1_epi32(static_cast<int>(i * n));
                __m256i resultIndices = _mm256_add_epi32(rowOffset, vecIndices);

                // Manually scatter-add the results
                alignas(32) int indicesArray[SIMD_WIDTH];
                alignas(32) int valuesArray[SIMD_WIDTH];
                _mm256_storeu_si256((__m256i*)indicesArray, resultIndices);
                _mm256_storeu_si256((__m256i*)valuesArray, vecMul);

                for (int l = 0; l < SIMD_WIDTH; ++l) {
                    int idx = indicesArray[l];
                    int val = valuesArray[l];
                    result.denseData[idx] += val;
                }

                idxB += SIMD_WIDTH;
            }

            // Handle remaining elements
            for (; idxB < rowEndB; ++idxB) {
                size_t j = matB.colIdx[idxB];
                int valB = matB.values[idxB];
                result.denseData[i * n + j] += valA * valB;
            }
        }
    }

    return result;
}

Matrix multiplyMSO(Matrix& matA, Matrix& matB, size_t threadNum) {
    // Compress matrices if not already compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Initialize result.denseData to zero
    result.denseData.resize(n * n);
    std::fill(result.denseData.begin(), result.denseData.end(), 0);

    const size_t SIMD_WIDTH = 8; // AVX2 processes 8 integers

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        // Local buffer to accumulate results before writing to shared memory
        std::vector<int> localResult(n, 0);

        size_t rowStartA = matA.rowPtr[i];
        size_t rowEndA = matA.rowPtr[i + 1];

        // Loop over non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            size_t k = matA.colIdx[idxA];  // Column index in matA
            int valA = matA.values[idxA];  // Non-zero value in matA

            // Get the row k of matB
            size_t rowStartB = matB.rowPtr[k];
            size_t rowEndB = matB.rowPtr[k + 1];

            size_t idxB = rowStartB;

            // Process elements in chunks of SIMD_WIDTH
            while (idxB + SIMD_WIDTH <= rowEndB) {
                // Load column indices and values from matB
                __m256i vecIndices = _mm256_loadu_si256((__m256i*)&matB.colIdx[idxB]);
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.values[idxB]);

                // Multiply valA with vecB
                __m256i vecA = _mm256_set1_epi32(valA);
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);

                // Extract indices and values
                alignas(32) int indicesArray[SIMD_WIDTH];
                alignas(32) int valuesArray[SIMD_WIDTH];
                _mm256_storeu_si256((__m256i*)indicesArray, vecIndices);
                _mm256_storeu_si256((__m256i*)valuesArray, vecMul);

                // Accumulate results in the local buffer
                for (int l = 0; l < SIMD_WIDTH; ++l) {
                    size_t j = indicesArray[l];
                    int val = valuesArray[l];
                    localResult[j] += val;
                }

                idxB += SIMD_WIDTH;
            }

            // Handle remaining elements
            for (; idxB < rowEndB; ++idxB) {
                size_t j = matB.colIdx[idxB];
                int valB = matB.values[idxB];
                localResult[j] += valA * valB;
            }
        }

        // Write local results to the shared result matrix
        for (size_t j = 0; j < n; ++j) {
            if (localResult[j] != 0) {
                // Use atomic operation to prevent data races
                #pragma omp atomic
                result.denseData[i * n + j] += localResult[j];
            }
        }
    }

    return result;
}
