
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

    if (type == DENSE) {
        // Calculate the total number of non-zero elements needed (95% of n^2)
        size_t totalElements = n * n;
        size_t nnzTotal = static_cast<size_t>(0.95 * totalElements);
        size_t numZeros = totalElements - nnzTotal;

        // Initialize all elements to non-zero values
        for (size_t i = 0; i < n; ++i) {
            rows[i].reserve(n);
            for (size_t j = 0; j < n; ++j) {
                int value = std::rand() % 10 + 1; // Random values between 1 and 10
                rows[i].emplace_back(j, value);
            }
            if (rows[i].size() > maxNNZ) {
                maxNNZ = rows[i].size();
            }
        }

        // Randomly assign zeros
        for (size_t idx = 0; idx < numZeros; ++idx) {
            size_t i = std::rand() % n;
            size_t j = std::rand() % n;

            // Find and remove the element to set it to zero
            auto& row = rows[i];
            bool found = false;
            for (auto it = row.begin(); it != row.end(); ++it) {
                if (it->first == j) {
                    row.erase(it);
                    found = true;
                    break;
                }
            }

            if (!found) {
                // If the element was already zero (unlikely), retry
                idx--;
            }
        }

        // Update maxNNZPerRow after removing zeros
        maxNNZ = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t nnzInRow = rows[i].size();
            if (nnzInRow > maxNNZ) {
                maxNNZ = nnzInRow;
            }
        }

    } else {
        // For sparse matrices, O(n) non-zero elements
        size_t nnzTotal = n; // Total number of non-zero elements

        // Assign one non-zero element per row
        for (size_t i = 0; i < n; ++i) {
            size_t j = std::rand() % n;
            int value = std::rand() % 10 + 1; // Random values between 1 and 10
            rows[i].emplace_back(j, value);
            if (rows[i].size() > maxNNZ) {
                maxNNZ = rows[i].size();
            }
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
        // Remaining positions are already zero-initialized
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
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    for (size_t i = 0; i < n; ++i) {
        // Map to store column indices and their summed values
        std::unordered_map<size_t, int> rowValues;

        for (size_t k = 0; k < n; ++k) {
            int valA = matA.getElement(i, k);
            if (valA == 0) continue;

            for (size_t idxB = 0; idxB < matB.maxNNZPerRow; ++idxB) {
                size_t offsetB = k * matB.maxNNZPerRow + idxB;
                int valB = matB.values[offsetB];
                size_t colB = matB.colIndices[offsetB];
                if (valB == 0) continue;

                rowValues[colB] += valA * valB;
            }
        }

        // Convert map to vector
        for (const auto& pair : rowValues) {
            resultRows[i].emplace_back(pair.first, pair.second);
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        size_t offset = i * result.maxNNZPerRow;
        size_t k = 0;

        for (const auto& elem : resultRows[i]) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}



Matrix 
multiplyM(const Matrix& matA, const Matrix& matB, size_t threadNum) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        std::unordered_map<size_t, int> rowValues;

        for (size_t k = 0; k < n; ++k) {
            int valA = matA.getElement(i, k);
            if (valA == 0) continue;

            for (size_t idxB = 0; idxB < matB.maxNNZPerRow; ++idxB) {
                size_t offsetB = k * matB.maxNNZPerRow + idxB;
                int valB = matB.values[offsetB];
                size_t colB = matB.colIndices[offsetB];
                if (valB == 0) continue;

                #pragma omp atomic
                rowValues[colB] += valA * valB;
            }
        }

        // Convert map to vector
        for (const auto& pair : rowValues) {
            resultRows[i].emplace_back(pair.first, pair.second);
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        size_t offset = i * result.maxNNZPerRow;
        size_t k = 0;

        for (const auto& elem : resultRows[i]) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}



#include <immintrin.h>

Matrix 
multiplyS(const Matrix& matA, const Matrix& matB) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Determine maxNNZPerRow for result
    result.maxNNZPerRow = matB.maxNNZPerRow;
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    const size_t SIMD_WIDTH = 8;

    for (size_t i = 0; i < n; ++i) {
        size_t offsetA = i * matA.maxNNZPerRow;

        for (size_t idxA = 0; idxA < matA.maxNNZPerRow; ++idxA) {
            int valA = matA.values[offsetA + idxA];
            size_t colA = matA.colIndices[offsetA + idxA];
            if (valA == 0) continue;

            __m256i vecA = _mm256_set1_epi32(valA);

            size_t offsetB = colA * matB.maxNNZPerRow;

            size_t idxB = 0;
            for (; idxB + SIMD_WIDTH <= matB.maxNNZPerRow; idxB += SIMD_WIDTH) {
                // Load values from matB
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.values[offsetB + idxB]);
                __m256i vecCols = _mm256_loadu_si256((__m256i*)&matB.colIndices[offsetB + idxB]);

                // Multiply
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);

                // Load existing values from result
                size_t offsetR = i * result.maxNNZPerRow + idxB;
                __m256i vecRes = _mm256_loadu_si256((__m256i*)&result.values[offsetR]);

                // Add the multiplication result
                vecRes = _mm256_add_epi32(vecRes, vecMul);

                // Store back to result
                _mm256_storeu_si256((__m256i*)&result.values[offsetR], vecRes);

                // Copy column indices
                _mm256_storeu_si256((__m256i*)&result.colIndices[offsetR], vecCols);
            }

            // Handle remaining elements
            for (; idxB < matB.maxNNZPerRow; ++idxB) {
                int valB = matB.values[offsetB + idxB];
                size_t colB = matB.colIndices[offsetB + idxB];
                if (valB == 0) continue;

                size_t offsetR = i * result.maxNNZPerRow + idxB;
                result.values[offsetR] += valA * valB;
                result.colIndices[offsetR] = colB;
            }
        }
    }

    return result;
}

Matrix 
multiplyO(Matrix& matA, Matrix& matB) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::unordered_map<size_t, int>> resultRows(n);

    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                int valB = matB.csrValues[idxB];
                size_t colB = matB.csrColIdx[idxB];

                resultRows[i][colB] += valA * valB;
            }
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& rowMap : resultRows) {
        if (rowMap.size() > maxNNZ) {
            maxNNZ = rowMap.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        const auto& rowMap = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        for (const auto& elem : rowMap) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}


Matrix 
multiplyMS(const Matrix& matA, const Matrix& matB, size_t threadNum) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Determine maxNNZPerRow for result
    result.maxNNZPerRow = matB.maxNNZPerRow;
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    const size_t SIMD_WIDTH = 8;

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        size_t offsetA = i * matA.maxNNZPerRow;

        for (size_t idxA = 0; idxA < matA.maxNNZPerRow; ++idxA) {
            int valA = matA.values[offsetA + idxA];
            size_t colA = matA.colIndices[offsetA + idxA];
            if (valA == 0) continue;

            __m256i vecA = _mm256_set1_epi32(valA);

            size_t offsetB = colA * matB.maxNNZPerRow;

            size_t idxB = 0;
            for (; idxB + SIMD_WIDTH <= matB.maxNNZPerRow; idxB += SIMD_WIDTH) {
                __m256i vecB = _mm256_loadu_si256((__m256i*)&matB.values[offsetB + idxB]);
                __m256i vecCols = _mm256_loadu_si256((__m256i*)&matB.colIndices[offsetB + idxB]);

                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);

                size_t offsetR = i * result.maxNNZPerRow + idxB;
                __m256i vecRes = _mm256_loadu_si256((__m256i*)&result.values[offsetR]);

                vecRes = _mm256_add_epi32(vecRes, vecMul);

                _mm256_storeu_si256((__m256i*)&result.values[offsetR], vecRes);
                _mm256_storeu_si256((__m256i*)&result.colIndices[offsetR], vecCols);
            }

            // Handle remaining elements
            for (; idxB < matB.maxNNZPerRow; ++idxB) {
                int valB = matB.values[offsetB + idxB];
                size_t colB = matB.colIndices[offsetB + idxB];
                if (valB == 0) continue;

                size_t offsetR = i * result.maxNNZPerRow + idxB;
                #pragma omp atomic
                result.values[offsetR] += valA * valB;
                result.colIndices[offsetR] = colB;
            }
        }
    }

    return result;
}


Matrix 
multiplyMO(Matrix& matA, Matrix& matB, size_t threadNum) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::unordered_map<size_t, int>> resultRows(n);

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        auto& rowMap = resultRows[i];

        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                int valB = matB.csrValues[idxB];
                size_t colB = matB.csrColIdx[idxB];

                #pragma omp atomic
                rowMap[colB] += valA * valB;
            }
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& rowMap : resultRows) {
        if (rowMap.size() > maxNNZ) {
            maxNNZ = rowMap.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        const auto& rowMap = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        for (const auto& elem : rowMap) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}

Matrix 
multiplySO(Matrix& matA, Matrix& matB) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::unordered_map<size_t, int>> resultRows(n);

    const size_t SIMD_WIDTH = 8;

    for (size_t i = 0; i < n; ++i) {
        auto& rowMap = resultRows[i];

        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            size_t lenB = rowEndB - rowStartB;
            size_t idxB = 0;

            // Process in SIMD_WIDTH chunks
            while (idxB + SIMD_WIDTH <= lenB) {
                // Load values and columns from matB
                __m256i vecValB = _mm256_loadu_si256((__m256i*)&matB.csrValues[rowStartB + idxB]);
                __m256i vecColB = _mm256_loadu_si256((__m256i*)&matB.csrColIdx[rowStartB + idxB]);

                // Broadcast valA
                __m256i vecValA = _mm256_set1_epi32(valA);

                // Multiply
                __m256i vecMul = _mm256_mullo_epi32(vecValA, vecValB);

                // Extract results and update rowMap
                alignas(32) int cols[SIMD_WIDTH];
                alignas(32) int vals[SIMD_WIDTH];
                _mm256_store_si256((__m256i*)cols, vecColB);
                _mm256_store_si256((__m256i*)vals, vecMul);

                for (size_t l = 0; l < SIMD_WIDTH; ++l) {
                    rowMap[cols[l]] += vals[l];
                }

                idxB += SIMD_WIDTH;
            }

            // Handle remaining elements
            for (; idxB < lenB; ++idxB) {
                int valB = matB.csrValues[rowStartB + idxB];
                size_t colB = matB.csrColIdx[rowStartB + idxB];

                rowMap[colB] += valA * valB;
            }
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& rowMap : resultRows) {
        if (rowMap.size() > maxNNZ) {
            maxNNZ = rowMap.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        const auto& rowMap = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        for (const auto& elem : rowMap) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}

Matrix 
multiplyMSO(Matrix& matA, Matrix& matB, size_t threadNum) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE);

    // Temporary storage for result rows
    std::vector<std::unordered_map<size_t, int>> resultRows(n);

    const size_t SIMD_WIDTH = 8;

    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        auto& rowMap = resultRows[i];

        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            size_t lenB = rowEndB - rowStartB;
            size_t idxB = 0;

            // Process in SIMD_WIDTH chunks
            while (idxB + SIMD_WIDTH <= lenB) {
                __m256i vecValB = _mm256_loadu_si256((__m256i*)&matB.csrValues[rowStartB + idxB]);
                __m256i vecColB = _mm256_loadu_si256((__m256i*)&matB.csrColIdx[rowStartB + idxB]);

                __m256i vecValA = _mm256_set1_epi32(valA);
                __m256i vecMul = _mm256_mullo_epi32(vecValA, vecValB);

                alignas(32) int cols[SIMD_WIDTH];
                alignas(32) int vals[SIMD_WIDTH];
                _mm256_store_si256((__m256i*)cols, vecColB);
                _mm256_store_si256((__m256i*)vals, vecMul);

                for (size_t l = 0; l < SIMD_WIDTH; ++l) {
                    #pragma omp atomic
                    rowMap[cols[l]] += vals[l];
                }

                idxB += SIMD_WIDTH;
            }

            // Handle remaining elements
            for (; idxB < lenB; ++idxB) {
                int valB = matB.csrValues[rowStartB + idxB];
                size_t colB = matB.csrColIdx[rowStartB + idxB];

                #pragma omp atomic
                rowMap[colB] += valA * valB;
            }
        }
    }

    // Determine maxNNZPerRow for result
    size_t maxNNZ = 0;
    for (const auto& rowMap : resultRows) {
        if (rowMap.size() > maxNNZ) {
            maxNNZ = rowMap.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize result ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Fill the result ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        const auto& rowMap = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        for (const auto& elem : rowMap) {
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}