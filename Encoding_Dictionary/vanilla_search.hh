// vanilla_search.hh
#pragma once

#include <vector>
#include <string>
#include <utility>
#include <thread>            // Required for std::thread and hardware_concurrency

// Forward declaration (if needed)
// class VanillaSearch;

class VanillaSearch {
public:
    /**
     * @brief Constructs a VanillaSearch object.
     * 
     * @param raw_data Reference to a vector of raw strings to search.
     * @param thread_num Number of threads to use for searching. Defaults to the number of hardware threads.
     */
    VanillaSearch(const std::vector<std::string>& raw_data, size_t thread_num = std::thread::hardware_concurrency());
    
    /**
     * @brief Performs an exact match search for the given query.
     * 
     * @param query The string to search for.
     * @param indices Vector to store the indices where the query is found.
     * @return true If the query is found at least once.
     * @return false If the query is not found.
     */
    bool exact_match(const std::string& query, std::vector<size_t>& indices) const;
    
    /**
     * @brief Performs a prefix search for the given prefix.
     * 
     * @param prefix The prefix string to search for.
     * @return A vector of pairs, each containing a matching string and a vector of indices where it appears.
     */
    std::vector<std::pair<std::string, std::vector<size_t>>> prefix_search(const std::string& prefix) const;
    
private:
    /**
     * @brief Worker function for exact match search.
     * 
     * @param query The string to search for.
     * @param start The starting index of the data chunk.
     * @param end The ending index of the data chunk.
     * @param local_indices Reference to a vector to store found indices in this chunk.
     */
    void exact_match_worker(const std::string& query, size_t start, size_t end, std::vector<size_t>& local_indices) const;
    
    /**
     * @brief Worker function for prefix search.
     * 
     * @param prefix The prefix string to search for.
     * @param prefix_length The length of the prefix to optimize comparisons.
     * @param start The starting index of the data chunk.
     * @param end The ending index of the data chunk.
     * @param local_matches Reference to a vector to store found (string, index) pairs in this chunk.
     */
    void prefix_search_worker(const std::string& prefix, size_t prefix_length, size_t start, size_t end, std::vector<std::pair<std::string, size_t>>& local_matches) const;
    
    // Member Variables
    const std::vector<std::string>& raw_data_;  // Reference to the raw data to search
    size_t thread_num_;                        // Number of threads to utilize
};
