
#include "matrix_multi.hh"
#include <iostream>
#include <string>
#include <unordered_set>
#include <random>
#include <chrono>
#include <cstdlib>
#include <utility>
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
    : n(size), type(matrixType), maxNNZPerRow(0), multiplicationCount(0), additionCount(0), compressed(false) {
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



//Get an element from the matrix
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


void 
Matrix::generateRandomData() 
{
    // Initialize random number generators
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_int_distribution<size_t> dist_col(0, n - 1);
    std::uniform_int_distribution<int> dist_val(1, 10); // Non-zero values between 1 and 10

    // Temporary storage for row-wise data
    std::vector<std::vector<std::pair<size_t, int>>> rows(n);

    if (type == DENSE) {
        // Define sparsity for dense matrix: 0.1% zeros
        double zero_fraction = 0.001; // 0.1%
        size_t total_elements = n * n;
        size_t num_zeros = static_cast<size_t>(zero_fraction * total_elements);

        // Initialize all elements as non-zero
        for (size_t i = 0; i < n; ++i) {
            rows[i].reserve(n);
            for (size_t j = 0; j < n; ++j) {
                int value = dist_val(rng);
                rows[i].emplace_back(j, value);
            }
        }

        // Randomly assign zeros
        // Generate unique zero positions
        std::unordered_set<size_t> zero_positions;
        while (zero_positions.size() < num_zeros) {
            size_t pos = dist_col(rng) + dist_col(rng) * n; // Row-major order
            zero_positions.insert(pos);
        }

        // Set selected positions to zero by removing them from rows
        for (const auto& pos : zero_positions) {
            size_t row = pos / n;
            size_t col = pos % n;

            auto& current_row = rows[row];
            // Find the element with column 'col' and remove it
            auto it = std::find_if(current_row.begin(), current_row.end(),
                                   [col](const std::pair<size_t, int>& elem) { return elem.first == col; });
            if (it != current_row.end()) {
                current_row.erase(it);
            }
        }

        // Update maxNNZPerRow after removing zeros
        maxNNZPerRow = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t nnzInRow = rows[i].size();
            if (nnzInRow > maxNNZPerRow) {
                maxNNZPerRow = nnzInRow;
            }
        }

    } else { // SPARSE
        // Define sparsity for sparse matrix: 1% zeros
        double zero_fraction = 0.01; // 1%
        size_t total_elements = n * n;
        size_t num_zeros = static_cast<size_t>(zero_fraction * total_elements);
        size_t num_nonzeros = total_elements - num_zeros;

        // Initialize all elements as non-zero
        for (size_t i = 0; i < n; ++i) {
            rows[i].reserve(n);
            for (size_t j = 0; j < n; ++j) {
                int value = dist_val(rng);
                rows[i].emplace_back(j, value);
            }
        }

        // Randomly assign zeros
        // Generate unique zero positions
        std::unordered_set<size_t> zero_positions;
        while (zero_positions.size() < num_zeros) {
            size_t pos = dist_col(rng) + dist_col(rng) * n; // Row-major order
            zero_positions.insert(pos);
        }

        // Set selected positions to zero by removing them from rows
        for (const auto& pos : zero_positions) {
            size_t row = pos / n;
            size_t col = pos % n;

            auto& current_row = rows[row];
            // Find the element with column 'col' and remove it
            auto it = std::find_if(current_row.begin(), current_row.end(),
                                   [col](const std::pair<size_t, int>& elem) { return elem.first == col; });
            if (it != current_row.end()) {
                current_row.erase(it);
            }
        }

        // Update maxNNZPerRow after removing zeros
        maxNNZPerRow = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t nnzInRow = rows[i].size();
            if (nnzInRow > maxNNZPerRow) {
                maxNNZPerRow = nnzInRow;
            }
        }
    }

    // Resize the values and colIndices arrays based on maxNNZPerRow
    values.resize(n * maxNNZPerRow, 0);
    colIndices.resize(n * maxNNZPerRow, 0);

    // Fill the ELLPACK structures
    for (size_t i = 0; i < n; ++i) {
        size_t offset = i * maxNNZPerRow;
        size_t k = 0;

        for (const auto& elem : rows[i]) {
            if (k < maxNNZPerRow) {
                values[offset + k] = elem.second;
                colIndices[offset + k] = elem.first;
                ++k;
            } else {
                // This should not happen as maxNNZPerRow is the maximum across all rows
                std::cerr << "Error: Exceeded maxNNZPerRow while filling ELLPACK structures." << std::endl;
                break;
            }
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

    // Ensure that csrRowPtr has size n + 1
    if (csrRowPtr.size() != n + 1) {
        std::cerr << "Error: csrRowPtr size mismatch after compression." << std::endl;
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
    result.multiplicationCount = 0;
    result.additionCount = 0;

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
                result.multiplicationCount += 1;
                result.additionCount += 1;
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

Matrix multiplyM(const Matrix& matA, const Matrix& matB, size_t threadNum) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);
    result.multiplicationCount = 0;
    result.additionCount = 0;

    // Temporary storage for result rows
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    // Initialize local counters for multiplication and addition
    size_t localMultCount = 0;
    size_t localAddCount = 0;

    // Parallelize the outer loop over rows using OpenMP with reduction for counters
    #pragma omp parallel for num_threads(threadNum) reduction(+:localMultCount, localAddCount)
    for (size_t i = 0; i < n; ++i) {
        std::unordered_map<size_t, int> rowValues;

        for (size_t k = 0; k < n; ++k) {
            int valA = matA.getElement(i, k);
            if (valA == 0) continue;

            for (size_t idxB = 0; idxB < matB.maxNNZPerRow; ++idxB) {
                size_t offsetB = k * matB.maxNNZPerRow + idxB;
                if (offsetB >= matB.values.size() || offsetB >= matB.colIndices.size()) {
                    std::cerr << "Error: ELLPACK index out of bounds in matB." << std::endl;
                    continue;
                }

                int valB = matB.values[offsetB];
                size_t colB = matB.colIndices[offsetB];
                if (valB == 0) continue;

                // Accumulate the multiplication result
                int product = valA * valB;
                // Increment multiplication counter
                localMultCount += 1;

                // Check if this is the first addition to this column
                if (rowValues.find(colB) == rowValues.end()) {
                    rowValues[colB] = product;
                } else {
                    rowValues[colB] += product;
                    // Increment addition counter
                    localAddCount += 1;
                }
            }
        }

        // Convert map to vector
        for (const auto& pair : rowValues) {
            resultRows[i].emplace_back(pair.first, pair.second);
        }
    } // End of parallel region

    // Assign the accumulated counts to the result matrix
    result.multiplicationCount = localMultCount;
    result.additionCount = localAddCount;

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
            if (k >= result.maxNNZPerRow) break; // Prevent overflow
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}

Matrix multiplyS(const Matrix& matA, const Matrix& matB) 
{
    size_t n = matA.n;
    Matrix result(n, DENSE);

    result.multiplicationCount = 0;
    result.additionCount = 0;

    // Determine maxNNZPerRow for result
    result.maxNNZPerRow = matB.maxNNZPerRow;
    result.values.resize(n * result.maxNNZPerRow, 0);
    result.colIndices.resize(n * result.maxNNZPerRow, 0);

    const size_t SIMD_WIDTH = 8; // AVX2 processes 8 integers at a time

    // Initialize local counters for multiplication and addition
    size_t localMultCount = 0;
    size_t localAddCount = 0;

    for (size_t i = 0; i < n; ++i) {
        size_t offsetA = i * matA.maxNNZPerRow;

        for (size_t idxA = 0; idxA < matA.maxNNZPerRow; ++idxA) {
            int valA = matA.values[offsetA + idxA];
            size_t colA = matA.colIndices[offsetA + idxA];
            if (valA == 0) continue;

            // Broadcast valA across SIMD registers
            __m256i vecA = _mm256_set1_epi32(valA);

            size_t offsetB = colA * matB.maxNNZPerRow;

            size_t idxB = 0;
            for (; idxB + SIMD_WIDTH <= matB.maxNNZPerRow; idxB += SIMD_WIDTH) {
                // Load values from matB
                __m256i vecB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.values[offsetB + idxB]));
                __m256i vecCols = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.colIndices[offsetB + idxB]));

                // Multiply
                __m256i vecMul = _mm256_mullo_epi32(vecA, vecB);
                localMultCount += SIMD_WIDTH; // Each element in SIMD vector represents a multiplication

                // Load existing values from result
                size_t offsetR = i * result.maxNNZPerRow + idxB;
                __m256i vecRes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.values[offsetR]));

                // Add the multiplication result
                __m256i vecAdd = _mm256_add_epi32(vecRes, vecMul);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result.values[offsetR]), vecAdd);
                localAddCount += SIMD_WIDTH; // Each element in SIMD vector represents an addition

                // Copy column indices
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result.colIndices[offsetR]), vecCols);
            }

            // Handle remaining elements
            for (; idxB < matB.maxNNZPerRow; ++idxB) {
                int valB = matB.values[offsetB + idxB];
                size_t colB = matB.colIndices[offsetB + idxB];
                if (valB == 0) continue;

                size_t offsetR = i * result.maxNNZPerRow + idxB;
                result.values[offsetR] += valA * valB;
                result.colIndices[offsetR] = colB;

                // Increment counters for scalar operations
                localMultCount += 1;
                localAddCount += 1;
            }
        }
    }

    // Assign the accumulated counts to the result matrix
    result.multiplicationCount = localMultCount;
    result.additionCount = localAddCount;

    return result;
}

Matrix multiplyO(Matrix& matA, Matrix& matB) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE); // Result will be in ELLPACK format

    // Safeguard: Ensure matrices are compatible for multiplication
    if (matA.n != matB.n) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        return result; // Return an empty result matrix
    }

    // Safeguard: Ensure csrRowPtr sizes are correct
    if (matA.csrRowPtr.size() != n + 1 || matB.csrRowPtr.size() != n + 1) {
        std::cerr << "Error: csrRowPtr size does not match matrix dimensions." << std::endl;
        return result;
    }

    // Temporary storage for result rows using vectors instead of unordered_map
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    // Initialize local counters for multiplication and addition
    size_t localMultCount = 0;
    size_t localAddCount = 0;

    // Iterate over rows of matA
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        // Temporary accumulation array
        std::vector<int> tempRow(n, 0);
        std::vector<size_t> activeCols;

        // Process non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            // Safeguard: Check if idxA is within bounds
            if (idxA >= matA.csrValues.size() || idxA >= matA.csrColIdx.size()) {
                std::cerr << "Error: csrValues or csrColIdx index out of bounds in matA." << std::endl;
                continue; // Skip to next idxA
            }

            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            // Safeguard: Check if colA is a valid row index in matB
            if (colA >= matB.n) {
                std::cerr << "Error: Column index out of bounds in matA (colA)." << std::endl;
                continue; // Skip to next idxA
            }

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            // Process non-zero elements in row colA of matB
            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                // Safeguard: Check if idxB is within bounds
                if (idxB >= matB.csrValues.size() || idxB >= matB.csrColIdx.size()) {
                    std::cerr << "Error: csrValues or csrColIdx index out of bounds in matB." << std::endl;
                    continue; // Skip to next idxB
                }

                int valB = matB.csrValues[idxB];
                size_t colB = matB.csrColIdx[idxB];

                // Safeguard: Check if colB is valid
                if (colB >= n) {
                    std::cerr << "Error: Column index out of bounds in matB (colB)." << std::endl;
                    continue; // Skip to next idxB
                }

                // Accumulate the multiplication result
                if (tempRow[colB] == 0) {
                    activeCols.push_back(colB);
                } else {
                    // If tempRow[colB] is already non-zero, an addition will occur
                    localAddCount += 1;
                }
                tempRow[colB] += valA * valB;

                // Each valA * valB represents one multiplication
                localMultCount += 1;
            }
        }

        // Transfer non-zero elements to resultRows
        for (const auto& colB : activeCols) {
            if (tempRow[colB] != 0) {
                resultRows[i].emplace_back(colB, tempRow[colB]);
            }
        }
    }

    // Assign the accumulated counts to the result matrix
    result.multiplicationCount = localMultCount;
    result.additionCount = localAddCount;

    // Determine maxNNZPerRow for the result matrix
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize the result matrix's ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Populate the result matrix from the temporary resultRows map
    for (size_t i = 0; i < n; ++i) {
        const auto& row = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        // Optional: Sort the column indices for each row for consistency
        std::vector<std::pair<size_t, int>> sortedRow(row);
        std::sort(sortedRow.begin(), sortedRow.end());

        for (const auto& elem : sortedRow) {
            if (k >= maxNNZ) break; // Safeguard: Prevent overflow in ELLPACK structure
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
    result.multiplicationCount = 0;
    result.additionCount = 0;

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

                // #pragma omp atomic
                // result.multiplicationCount += SIMD_WIDTH; // 8 multiplications

                // #pragma omp atomic
                // result.additionCount += SIMD_WIDTH;
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

                //  #pragma omp atomic
                // result.multiplicationCount += 1; 

                // #pragma omp atomic
                // result.additionCount += 1;  
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
    Matrix result(n, DENSE); // Result will be in ELLPACK format

    // Safeguard: Ensure matrices are compatible for multiplication
    if (matA.n != matB.n) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        return result; // Return an empty result matrix
    }

    // Safeguard: Ensure csrRowPtr sizes are correct
    if (matA.csrRowPtr.size() != n + 1 || matB.csrRowPtr.size() != n + 1) {
        std::cerr << "Error: csrRowPtr size does not match matrix dimensions." << std::endl;
        return result;
    }

    // Temporary storage for result rows using vectors instead of unordered_map
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    // Parallelize the outer loop over rows using OpenMP
    #pragma omp parallel for num_threads(threadNum)
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        // Temporary accumulation array
        std::vector<int> tempRow(n, 0);
        std::vector<size_t> activeCols;

        // Process non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            // Safeguard: Check if idxA is within bounds
            if (idxA >= matA.csrValues.size() || idxA >= matA.csrColIdx.size()) {
                std::cerr << "Error: csrValues or csrColIdx index out of bounds in matA." << std::endl;
                continue; // Skip to next idxA
            }

            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            // Safeguard: Check if colA is a valid row index in matB
            if (colA >= matB.n) {
                std::cerr << "Error: Column index out of bounds in matA (colA)." << std::endl;
                continue; // Skip to next idxA
            }

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];

            // Process non-zero elements in row colA of matB
            for (size_t idxB = rowStartB; idxB < rowEndB; ++idxB) {
                // Safeguard: Check if idxB is within bounds
                if (idxB >= matB.csrValues.size() || idxB >= matB.csrColIdx.size()) {
                    std::cerr << "Error: csrValues or csrColIdx index out of bounds in matB." << std::endl;
                    continue; // Skip to next idxB
                }

                int valB = matB.csrValues[idxB];
                size_t colB = matB.csrColIdx[idxB];

                // Safeguard: Check if colB is valid
                if (colB >= n) {
                    std::cerr << "Error: Column index out of bounds in matB (colB)." << std::endl;
                    continue; // Skip to next idxB
                }

                // Accumulate the multiplication result
                if (tempRow[colB] == 0) {
                    activeCols.push_back(colB);
                }
                tempRow[colB] += valA * valB;
            }
        }

        // Transfer non-zero elements to resultRows
        for (const auto& colB : activeCols) {
            if (tempRow[colB] != 0) {
                resultRows[i].emplace_back(colB, tempRow[colB]);
            }
        }
    }

    // Determine maxNNZPerRow for the result matrix
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize the result matrix's ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Populate the result matrix from the temporary resultRows map
    for (size_t i = 0; i < n; ++i) {
        const auto& row = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        // Optional: Sort the column indices for each row for consistency
        std::vector<std::pair<size_t, int>> sortedRow(row);
        std::sort(sortedRow.begin(), sortedRow.end());

        for (const auto& elem : sortedRow) {
            if (k >= maxNNZ) break; // Safeguard: Prevent overflow in ELLPACK structure
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
    Matrix result(n, DENSE); // Result will be in ELLPACK format

    // Safeguard: Ensure matrices are compatible for multiplication
    if (matA.n != matB.n) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        return result; // Return an empty result matrix
    }

    // Safeguard: Ensure csrRowPtr sizes are correct
    if (matA.csrRowPtr.size() != n + 1 || matB.csrRowPtr.size() != n + 1) {
        std::cerr << "Error: csrRowPtr size does not match matrix dimensions." << std::endl;
        return result;
    }

    // Temporary storage for result rows using vectors instead of unordered_map
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    const size_t SIMD_WIDTH = 8; // AVX2 can process 8 integers at a time

    // Iterate over rows of matA
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        // Temporary accumulation array
        std::vector<int> tempRow(n, 0);
        std::vector<size_t> activeCols;

        // Process non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            // Safeguard: Check if idxA is within bounds
            if (idxA >= matA.csrValues.size() || idxA >= matA.csrColIdx.size()) {
                std::cerr << "Error: csrValues or csrColIdx index out of bounds in matA." << std::endl;
                continue; // Skip to next idxA
            }

            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            // Safeguard: Check if colA is a valid row index in matB
            if (colA >= matB.n) {
                std::cerr << "Error: Column index out of bounds in matA (colA)." << std::endl;
                continue; // Skip to next idxA
            }

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];
            size_t nnzB = rowEndB - rowStartB;

            size_t idxB = 0;

            // SIMD processing
            for (; idxB + SIMD_WIDTH <= nnzB; idxB += SIMD_WIDTH) {
                // Load values and column indices from matB
                __m256i vecValB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.csrValues[rowStartB + idxB]));
                __m256i vecColB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.csrColIdx[rowStartB + idxB]));

                // Broadcast valA
                __m256i vecValA = _mm256_set1_epi32(valA);

                // Multiply
                __m256i vecMul = _mm256_mullo_epi32(vecValA, vecValB);

                // Store results
                alignas(32) int cols[SIMD_WIDTH];
                alignas(32) int vals[SIMD_WIDTH];
                _mm256_store_si256(reinterpret_cast<__m256i*>(cols), vecColB);
                _mm256_store_si256(reinterpret_cast<__m256i*>(vals), vecMul);

                // Accumulate results
                for (size_t l = 0; l < SIMD_WIDTH; ++l) {
                    size_t colB = static_cast<size_t>(cols[l]);
                    int val = vals[l];
                    if (colB >= n) {
                        std::cerr << "Error: Column index out of bounds in matB (colB) during SIMD." << std::endl;
                        continue;
                    }
                    if (tempRow[colB] == 0) {
                        activeCols.push_back(colB);
                    }
                    tempRow[colB] += val;
                }
            }

            // Handle remaining elements
            for (; idxB < nnzB; ++idxB) {
                // Safeguard: Check if idxB is within bounds
                if (rowStartB + idxB >= matB.csrValues.size() || rowStartB + idxB >= matB.csrColIdx.size()) {
                    std::cerr << "Error: csrValues or csrColIdx index out of bounds in matB." << std::endl;
                    continue;
                }

                int valB = matB.csrValues[rowStartB + idxB];
                size_t colB = matB.csrColIdx[rowStartB + idxB];

                if (colB >= n) {
                    std::cerr << "Error: Column index out of bounds in matB (colB)." << std::endl;
                    continue;
                }

                // Accumulate the multiplication result
                if (tempRow[colB] == 0) {
                    activeCols.push_back(colB);
                }
                tempRow[colB] += valA * valB;
            }
        }

        // Transfer non-zero elements to resultRows
        for (const auto& colB : activeCols) {
            if (tempRow[colB] != 0) {
                resultRows[i].emplace_back(colB, tempRow[colB]);
            }
        }
    }

    // Determine maxNNZPerRow for the result matrix
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize the result matrix's ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Populate the result matrix from the temporary resultRows map
    for (size_t i = 0; i < n; ++i) {
        const auto& row = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        // Optional: Sort the column indices for each row for consistency
        std::vector<std::pair<size_t, int>> sortedRow(row);
        std::sort(sortedRow.begin(), sortedRow.end());

        for (const auto& elem : sortedRow) {
            if (k >= maxNNZ) break; // Safeguard: Prevent overflow in ELLPACK structure
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}

Matrix multiplyMSO(Matrix& matA, Matrix& matB, size_t threadNum) 
{
    // Ensure matrices are compressed
    if (!matA.compressed) matA.compress();
    if (!matB.compressed) matB.compress();

    size_t n = matA.n;
    Matrix result(n, DENSE); // Result will be in ELLPACK format

    // Safeguard: Ensure matrices are compatible for multiplication
    if (matA.n != matB.n) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        return result; // Return an empty result matrix
    }

    // Safeguard: Ensure csrRowPtr sizes are correct
    if (matA.csrRowPtr.size() != n + 1 || matB.csrRowPtr.size() != n + 1) {
        std::cerr << "Error: csrRowPtr size does not match matrix dimensions." << std::endl;
        return result;
    }

    // Temporary storage for result rows using vectors instead of unordered_map
    std::vector<std::vector<std::pair<size_t, int>>> resultRows(n);

    const size_t SIMD_WIDTH = 8; // AVX2 can process 8 integers at a time

    // Initialize local counters for multiplication and addition
    size_t localMultCount = 0;
    size_t localAddCount = 0;

    // Parallelize the outer loop over rows using OpenMP with reduction for counters
    #pragma omp parallel for num_threads(threadNum) reduction(+:localMultCount, localAddCount)
    for (size_t i = 0; i < n; ++i) {
        size_t rowStartA = matA.csrRowPtr[i];
        size_t rowEndA = matA.csrRowPtr[i + 1];

        // Temporary accumulation array
        std::vector<int> tempRow(n, 0);
        std::vector<size_t> activeCols;

        // Process non-zero elements in row i of matA
        for (size_t idxA = rowStartA; idxA < rowEndA; ++idxA) {
            // Safeguard: Check if idxA is within bounds
            if (idxA >= matA.csrValues.size() || idxA >= matA.csrColIdx.size()) {
                std::cerr << "Error: csrValues or csrColIdx index out of bounds in matA." << std::endl;
                continue; // Skip to next idxA
            }

            int valA = matA.csrValues[idxA];
            size_t colA = matA.csrColIdx[idxA];

            // Safeguard: Check if colA is a valid row index in matB
            if (colA >= matB.n) {
                std::cerr << "Error: Column index out of bounds in matA (colA)." << std::endl;
                continue; // Skip to next idxA
            }

            size_t rowStartB = matB.csrRowPtr[colA];
            size_t rowEndB = matB.csrRowPtr[colA + 1];
            size_t nnzB = rowEndB - rowStartB;

            size_t idxB = 0;

            // SIMD processing
            for (; idxB + SIMD_WIDTH <= nnzB; idxB += SIMD_WIDTH) {
                // Load values and column indices from matB
                __m256i vecValB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.csrValues[rowStartB + idxB]));
                __m256i vecColB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&matB.csrColIdx[rowStartB + idxB]));

                // Broadcast valA
                __m256i vecValA = _mm256_set1_epi32(valA);

                // Multiply
                __m256i vecMul = _mm256_mullo_epi32(vecValA, vecValB);

                // Store results
                alignas(32) int cols[SIMD_WIDTH];
                alignas(32) int vals[SIMD_WIDTH];
                _mm256_store_si256(reinterpret_cast<__m256i*>(cols), vecColB);
                _mm256_store_si256(reinterpret_cast<__m256i*>(vals), vecMul);

                // Accumulate results
                for (size_t l = 0; l < SIMD_WIDTH; ++l) {
                    size_t colB = static_cast<size_t>(cols[l]);
                    int val = vals[l];
                    if (colB >= n) {
                        std::cerr << "Error: Column index out of bounds in matB (colB) during SIMD." << std::endl;
                        continue;
                    }
                    if (tempRow[colB] == 0) {
                        activeCols.push_back(colB);
                    }
                    tempRow[colB] += val;
                    
                    // Increment operation counters
                    localMultCount += 1; // Each SIMD multiply represents SIMD_WIDTH multiplications
                    localAddCount += 1;  // Each SIMD addition represents SIMD_WIDTH additions
                }
            }

            // Handle remaining elements
            for (; idxB < nnzB; ++idxB) {
                // Safeguard: Check if idxB is within bounds
                if (rowStartB + idxB >= matB.csrValues.size() || rowStartB + idxB >= matB.csrColIdx.size()) {
                    std::cerr << "Error: csrValues or csrColIdx index out of bounds in matB." << std::endl;
                    continue;
                }

                int valB = matB.csrValues[rowStartB + idxB];
                size_t colB = matB.csrColIdx[rowStartB + idxB];

                if (colB >= n) {
                    std::cerr << "Error: Column index out of bounds in matB (colB)." << std::endl;
                    continue;
                }

                // Accumulate the multiplication result
                if (tempRow[colB] == 0) {
                    activeCols.push_back(colB);
                }
                tempRow[colB] += valA * valB;
                
                // Increment operation counters
                localMultCount += 1;
                localAddCount += 1;
            }
        }

        // Transfer non-zero elements to resultRows
        for (const auto& colB : activeCols) {
            if (tempRow[colB] != 0) {
                resultRows[i].emplace_back(colB, tempRow[colB]);
            }
        }
    } // End of parallel region

    // Assign the accumulated counts to the result matrix
    result.multiplicationCount = localMultCount;
    result.additionCount = localAddCount;

    // Determine maxNNZPerRow for the result matrix
    size_t maxNNZ = 0;
    for (const auto& row : resultRows) {
        if (row.size() > maxNNZ) {
            maxNNZ = row.size();
        }
    }
    result.maxNNZPerRow = maxNNZ;

    // Initialize the result matrix's ELLPACK structures
    result.values.resize(n * maxNNZ, 0);
    result.colIndices.resize(n * maxNNZ, 0);

    // Populate the result matrix from the temporary resultRows map
    for (size_t i = 0; i < n; ++i) {
        const auto& row = resultRows[i];
        size_t offset = i * maxNNZ;
        size_t k = 0;

        // Optional: Sort the column indices for each row for consistency
        std::vector<std::pair<size_t, int>> sortedRow(row);
        std::sort(sortedRow.begin(), sortedRow.end());

        for (const auto& elem : sortedRow) {
            if (k >= maxNNZ) break; // Safeguard: Prevent overflow in ELLPACK structure
            result.values[offset + k] = elem.second;
            result.colIndices[offset + k] = elem.first;
            ++k;
        }
    }

    return result;
}
