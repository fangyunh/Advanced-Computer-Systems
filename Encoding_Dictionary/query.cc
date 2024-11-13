// query.cc
#include "query.hh"
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <stdexcept>
#include <cstring> // For std::memcmp

#ifdef __SSE4_2__
#include <nmmintrin.h> // SSE4.2 intrinsics
#endif

// Constructor with SIMD flag
QueryEngine::QueryEngine(bool use_simd)
    : use_simd_(use_simd) {}

// Load and deserialize the encoded column file
void QueryEngine::load_encoded_file(const std::string& filename) {
    try {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error("Failed to open encoded file for loading.");
        }

        // Read dictionary size
        uint32_t dict_size;
        ifs.read(reinterpret_cast<char*>(&dict_size), sizeof(dict_size));
        if (ifs.eof() || ifs.fail()) {
            throw std::runtime_error("Failed to read dictionary size.");
        }

        // Read dictionary entries
        dictionary_.clear();
        for (uint32_t i = 0; i < dict_size; ++i) {
            uint32_t key_size;
            ifs.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
            if (ifs.eof() || ifs.fail()) {
                throw std::runtime_error("Failed to read key size.");
            }

            std::string key(key_size, ' ');
            ifs.read(&key[0], key_size);
            if (ifs.eof() || ifs.fail()) {
                throw std::runtime_error("Failed to read key.");
            }

            uint32_t id;
            ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (ifs.eof() || ifs.fail()) {
                throw std::runtime_error("Failed to read ID.");
            }

            dictionary_[key] = id;
        }

        // Read encoded data size
        uint32_t data_size;
        ifs.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        if (ifs.eof() || ifs.fail()) {
            throw std::runtime_error("Failed to read encoded data size.");
        }

        // Read encoded data
        encoded_data_.resize(data_size);
        ifs.read(reinterpret_cast<char*>(encoded_data_.data()), data_size * sizeof(uint32_t));
        if (ifs.eof() || ifs.fail()) {
            throw std::runtime_error("Failed to read encoded data.");
        }

        // Build ID to indices map
        build_id_to_indices();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error during deserialization: " << ex.what() << std::endl;
        throw; // Re-throw after logging
    }
}

// Exact Match Query
bool QueryEngine::exact_match(const std::string& query, std::vector<size_t>& indices) const {
    auto it = dictionary_.find(query);
    if (it == dictionary_.end()) {
        return false; // Query string not found
    }

    uint32_t id = it->second;
    auto id_it = id_to_indices_.find(id);
    if (id_it != id_to_indices_.end()) {
        indices = id_it->second;
        return true;
    }

    return false;
}

// Prefix Search Query
std::vector<std::pair<std::string, std::vector<size_t>>> QueryEngine::prefix_search(const std::string& prefix) const {
    std::vector<std::pair<std::string, std::vector<size_t>>> results;
    if (prefix.empty()) {
        return results;
    }

    size_t prefix_length = prefix.size();

#ifdef __SSE4_2__
    if (use_simd_ && prefix_length <= 16) { // SSE register size
        // SIMD-enabled prefix comparison
        __m128i prefix_vector;
        char prefix_buffer[16] = {0};
        std::memcpy(prefix_buffer, prefix.c_str(), prefix_length);
        prefix_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prefix_buffer));

        for (const auto& [key, id] : dictionary_) {
            if (key.size() < prefix_length) continue;

            // Load key prefix into SIMD register
            char key_buffer[16] = {0};
            std::memcpy(key_buffer, key.c_str(), prefix_length);
            __m128i key_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key_buffer));

            // Compare the two vectors
            __m128i cmp = _mm_cmpeq_epi8(prefix_vector, key_vector);
            int mask = _mm_movemask_epi8(cmp);

            // Check if all prefix_length bytes are equal
            bool match = true;
            for (size_t i = 0; i < prefix_length; ++i) {
                if (((mask >> i) & 1) == 0) {
                    match = false;
                    break;
                }
            }

            if (match) {
                results.emplace_back(key, id_to_indices_.at(id));
            }
        }
    }
    else
#endif
    {
        // Fallback to scalar prefix comparison
        for (const auto& [key, id] : dictionary_) {
            if (key.size() >= prefix_length && key.compare(0, prefix_length, prefix) == 0) {
                results.emplace_back(key, id_to_indices_.at(id));
            }
        }
    }

    return results;
}

// Build ID to indices map using multi-threading
void QueryEngine::build_id_to_indices() {
    id_to_indices_.clear();

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback to 4 threads if detection fails
    size_t data_size = encoded_data_.size();
    size_t chunk_size = data_size / num_threads;

    // Each thread builds a local map
    std::vector<std::unordered_map<uint32_t, std::vector<size_t>>> local_maps(num_threads);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? data_size : start + chunk_size;
        threads.emplace_back([this, &local_maps, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                uint32_t id = encoded_data_[j];
                local_maps[i][id].push_back(j);
            }
        });
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Merge local maps into the global id_to_indices_ map
    for (const auto& local_map : local_maps) {
        for (const auto& [id, indices] : local_map) {
            auto& global_indices = id_to_indices_[id];
            global_indices.insert(global_indices.end(), indices.begin(), indices.end());
        }
    }
}

// SIMD-Optimized Prefix Comparison Helper Function
bool QueryEngine::simd_prefix_compare(const std::string& key, const std::string& prefix) const {
#ifdef __SSE4_2__
    if (!use_simd_ || prefix.size() > 16) {
        return false;
    }

    __m128i prefix_vector;
    char prefix_buffer[16] = {0};
    std::memcpy(prefix_buffer, prefix.c_str(), prefix.size());
    prefix_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(prefix_buffer));

    char key_buffer[16] = {0};
    std::memcpy(key_buffer, key.c_str(), prefix.size());
    __m128i key_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key_buffer));

    __m128i cmp = _mm_cmpeq_epi8(prefix_vector, key_vector);
    int mask = _mm_movemask_epi8(cmp);

    for (size_t i = 0; i < prefix.size(); ++i) {
        if (((mask >> i) & 1) == 0) {
            return false;
        }
    }

    return true;
#else
    return false;
#endif
}
