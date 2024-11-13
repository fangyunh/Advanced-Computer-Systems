// vanilla_search.cc
#include "vanilla_search.hh"
#include <thread>
#include <future>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

// Constructor with thread_num
VanillaSearch::VanillaSearch(const std::vector<std::string>& raw_data, size_t thread_num)
    : raw_data_(raw_data), thread_num_(thread_num) {}

// Exact Match Query
bool VanillaSearch::exact_match(const std::string& query, std::vector<size_t>& indices) const {
    indices.clear();
    size_t num_threads = std::min(thread_num_, raw_data_.size());
    size_t data_size = raw_data_.size();
    size_t chunk_size = data_size / num_threads;

    std::vector<std::vector<size_t>> thread_indices(num_threads);
    std::vector<std::thread> threads;

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? data_size : start + chunk_size;
        threads.emplace_back(&VanillaSearch::exact_match_worker, this, std::cref(query), start, end, std::ref(thread_indices[i]));
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Combine results
    for (const auto& local_idx : thread_indices) {
        indices.insert(indices.end(), local_idx.begin(), local_idx.end());
    }

    return !indices.empty();
}

// Exact Match Worker
void VanillaSearch::exact_match_worker(const std::string& query, size_t start, size_t end, std::vector<size_t>& local_indices) const {
    for (size_t i = start; i < end; ++i) {
        if (raw_data_[i] == query) {
            local_indices.push_back(i);
        }
    }
}

// Prefix Search Query
std::vector<std::pair<std::string, std::vector<size_t>>> VanillaSearch::prefix_search(const std::string& prefix) const {
    std::vector<std::pair<std::string, std::vector<size_t>>> results;
    if (prefix.empty()) {
        return results;
    }

    size_t prefix_length = prefix.size(); // Now used
    size_t num_threads = std::min(thread_num_, raw_data_.size());
    size_t data_size = raw_data_.size();
    size_t chunk_size = data_size / num_threads;

    // Each thread will collect local matches as (string, index)
    std::vector<std::vector<std::pair<std::string, size_t>>> thread_matches(num_threads);
    std::vector<std::thread> threads;

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? data_size : start + chunk_size;
        threads.emplace_back(&VanillaSearch::prefix_search_worker, this, std::cref(prefix), prefix_length, start, end, std::ref(thread_matches[i]));
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Merge local matches into a global map to ensure uniqueness
    std::unordered_map<std::string, std::vector<size_t>> global_map;
    for (const auto& local : thread_matches) {
        for (const auto& [str, idx] : local) {
            global_map[str].push_back(idx);
        }
    }

    // Populate the results vector
    for (const auto& [str, idxs] : global_map) {
        results.emplace_back(str, idxs);
    }

    return results;
}

// Prefix Search Worker
void VanillaSearch::prefix_search_worker(const std::string& prefix, size_t prefix_length, size_t start, size_t end, std::vector<std::pair<std::string, size_t>>& local_matches) const {
    for (size_t i = start; i < end; ++i) {
        const std::string& key = raw_data_[i];
        if (key.size() >= prefix_length && key.compare(0, prefix_length, prefix) == 0) {
            local_matches.emplace_back(key, i);
        }
    }
}
