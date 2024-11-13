// query.hh
#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>

class QueryEngine {
public:
    using DictMap = std::unordered_map<std::string, uint32_t>;

    // Constructor with SIMD flag
    explicit QueryEngine(bool use_simd = false);

    // Load and deserialize the encoded column file
    void load_encoded_file(const std::string& filename);

    // Exact Match Query:
    // Returns true if the query exists, and fills 'indices' with matching positions
    bool exact_match(const std::string& query, std::vector<size_t>& indices) const;

    // Prefix Search Query:
    // Returns a vector of pairs, each containing a matching string and its corresponding indices
    std::vector<std::pair<std::string, std::vector<size_t>>> prefix_search(const std::string& prefix) const;

private:
    DictMap dictionary_; // Maps strings to their unique IDs
    std::vector<uint32_t> encoded_data_; // Encoded data as dictionary IDs
    std::unordered_map<uint32_t, std::vector<size_t>> id_to_indices_; // Maps IDs to their indices in encoded_data_
    bool use_simd_; // Flag to indicate SIMD usage

    // Build the ID to indices map using multi-threading
    void build_id_to_indices();

    // Helper function to perform SIMD-optimized prefix comparison
    bool simd_prefix_compare(const std::string& key, const std::string& prefix) const;
};
