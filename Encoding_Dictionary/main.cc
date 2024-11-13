// main.cc
#include "encoder.hh"
#include "query.hh"
#include "vanilla_search.hh"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>    // For std::strcmp
#include <chrono>
#include <cstdlib>    // For std::atoi
#include <fstream>    // For std::ifstream
#include <thread>     // For std::thread::hardware_concurrency
#include <iomanip>    // For std::setprecision

int main(int argc, char* argv[]) {
    // Default parameters
    size_t thread_num = std::thread::hardware_concurrency();
    bool use_simd = false;
    bool use_vanilla = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                thread_num = std::atoi(argv[i + 1]);
                if (thread_num == 0) {
                    std::cerr << "Invalid thread number after -t.\n";
                    return 1;
                }
                ++i; // Skip the next argument as it's the thread number
            } else {
                std::cerr << "Missing thread number after -t.\n";
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-q") == 0) {
            use_simd = true;
        }
        else if (std::strcmp(argv[i], "-v") == 0) {
            use_vanilla = true;
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            return 1;
        }
    }

    // Check for conflicting arguments
    if (use_simd && use_vanilla) {
        std::cerr << "Arguments -q and -v cannot be used together.\n";
        return 1;
    }

    // Display the configuration
    std::cout << "Configuration:\n";
    std::cout << "Threads: " << thread_num << "\n";
    if (use_simd) {
        std::cout << "Query Mode: Dictionary Encoding with SIMD\n";
    }
    else if (use_vanilla) {
        std::cout << "Query Mode: Vanilla Search\n";
    }
    else {
        std::cout << "Query Mode: Dictionary Encoding without SIMD\n";
    }

    // File paths
    std::string raw_file = "Column.txt";
    std::string encoded_file = "encoded_column.bin";

    // Step 1: Load Raw Data into Memory (Not Timed)
    std::cout << "\nLoading raw data into memory...\n";
    std::ifstream raw_ifs(raw_file);
    if (!raw_ifs) {
        std::cerr << "Failed to open raw data file: " << raw_file << "\n";
        return 1;
    }

    std::vector<std::string> raw_data;
    raw_data.reserve(10000000); // Adjust based on expected data size

    std::string line;
    while (std::getline(raw_ifs, line)) {
        raw_data.emplace_back(line);
    }
    raw_ifs.close();
    std::cout << "Loaded " << raw_data.size() << " entries into memory.\n";

    // Step 2: Encode the Data (Timed)
    std::cout << "\nStarting encoding process...\n";
    DictionaryEncoder encoder;
    auto start_encode = std::chrono::high_resolution_clock::now();
    encoder.encode_from_memory(raw_data, thread_num);
    auto end_encode = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed_encode = end_encode - start_encode;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Encoding completed in " << elapsed_encode.count() / 1e9 << " seconds (" 
              << elapsed_encode.count() << " nanoseconds).\n";

    // Step 3: Serialize the Encoded Data (Not Timed)
    std::cout << "Serializing encoded data...\n";
    encoder.serialize(encoded_file);
    std::cout << "Serialization completed.\n";

    // Step 4: Initialize Query Engine or Vanilla Search
    if (use_vanilla) {
        // Initialize VanillaSearch
        std::cout << "\nInitializing VanillaSearch...\n";
        VanillaSearch vs(raw_data, thread_num);

        // Perform Sample Queries
        // Replace these with actual queries relevant to your data
        std::string exact_query = "phsdou";       // Example exact query
        std::string prefix_query = "phsva";       // Example prefix query

        // Configure output precision
        std::cout << std::fixed << std::setprecision(9);

        // Exact Match Query (Timed)
        std::cout << "\nPerforming Exact Match Query...\n";
        std::vector<size_t> vs_indices;
        auto start_vs_exact = std::chrono::high_resolution_clock::now();
        bool found_vs = vs.exact_match(exact_query, vs_indices);
        auto end_vs_exact = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed_vs_exact = end_vs_exact - start_vs_exact;

        std::cout << "\n=== VanillaSearch: Exact Match ===\n";
        if (found_vs) {
            std::cout << "Found '" << exact_query << "' at indices: ";
            for (const auto& idx : vs_indices) {
                std::cout << idx << " ";
            }
            std::cout << "\nTime Taken: " << elapsed_vs_exact.count() << " nanoseconds.\n";
        }
        else {
            std::cout << "'" << exact_query << "' not found.\n";
            std::cout << "Time Taken: " << elapsed_vs_exact.count() << " nanoseconds.\n";
        }

        // Prefix Search Query (Timed)
        std::cout << "\nPerforming Prefix Search Query...\n";
        auto start_vs_prefix = std::chrono::high_resolution_clock::now();
        auto vs_prefix_results = vs.prefix_search(prefix_query);
        auto end_vs_prefix = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed_vs_prefix = end_vs_prefix - start_vs_prefix;

        std::cout << "\n=== VanillaSearch: Prefix Search ===\n";
        if (!vs_prefix_results.empty()) {
            for (const auto& [str, idxs] : vs_prefix_results) {
                std::cout << "String: " << str << ", Indices: ";
                for (const auto& idx : idxs) {
                    std::cout << idx << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Time Taken: " << elapsed_vs_prefix.count() << " nanoseconds.\n";
        }
        else {
            std::cout << "No strings found with prefix '" << prefix_query << "'.\n";
            std::cout << "Time Taken: " << elapsed_vs_prefix.count() << " nanoseconds.\n";
        }
    }
    else {
        // Initialize QueryEngine with or without SIMD
        std::cout << "\nInitializing QueryEngine...\n";
        QueryEngine qe(use_simd);

        // Load the encoded data (Not Timed)
        std::cout << "Loading encoded data from '" << encoded_file << "'...\n";
        qe.load_encoded_file(encoded_file);
        std::cout << "Encoded data loaded.\n";

        // Perform Sample Queries
        // Replace these with actual queries relevant to your data
        std::string exact_query = "phsdou";       // Example exact query
        std::string prefix_query = "phsva";       // Example prefix query

        // Configure output precision
        std::cout << std::fixed << std::setprecision(9);

        // Exact Match Query (Timed)
        std::cout << "\nPerforming Exact Match Query...\n";
        std::vector<size_t> qe_indices;
        auto start_qe_exact = std::chrono::high_resolution_clock::now();
        bool found_qe = qe.exact_match(exact_query, qe_indices);
        auto end_qe_exact = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed_qe_exact = end_qe_exact - start_qe_exact;

        std::cout << "\n=== QueryEngine: Exact Match ===\n";
        if (found_qe) {
            std::cout << "Found '" << exact_query << "' at indices: ";
            for (const auto& idx : qe_indices) {
                std::cout << idx << " ";
            }
            std::cout << "\nTime Taken: " << elapsed_qe_exact.count() << " nanoseconds.\n";
        }
        else {
            std::cout << "'" << exact_query << "' not found.\n";
            std::cout << "Time Taken: " << elapsed_qe_exact.count() << " nanoseconds.\n";
        }

        // Prefix Search Query (Timed)
        std::cout << "\nPerforming Prefix Search Query...\n";
        auto start_qe_prefix = std::chrono::high_resolution_clock::now();
        auto qe_prefix_results = qe.prefix_search(prefix_query);
        auto end_qe_prefix = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> elapsed_qe_prefix = end_qe_prefix - start_qe_prefix;

        std::cout << "\n=== QueryEngine: Prefix Search ===\n";
        if (!qe_prefix_results.empty()) {
            for (const auto& [str, idxs] : qe_prefix_results) {
                std::cout << "String: " << str << ", Indices: ";
                for (const auto& idx : idxs) {
                    std::cout << idx << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Time Taken: " << elapsed_qe_prefix.count() << " nanoseconds.\n";
        }
        else {
            std::cout << "No strings found with prefix '" << prefix_query << "'.\n";
            std::cout << "Time Taken: " << elapsed_qe_prefix.count() << " nanoseconds.\n";
        }
    }

    return 0;
}
