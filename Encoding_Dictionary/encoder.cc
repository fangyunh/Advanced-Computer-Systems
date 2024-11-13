#include "encoder.hh"
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <stdexcept>

// Encode function: Reads, builds dictionary, and encodes data
void DictionaryEncoder::encode(const std::string& filename, size_t chunk_size_bytes, size_t thread_num) 
{
    try {
        // Step 1: Read file in chunks
        auto chunks = read_file_in_chunks(filename, chunk_size_bytes);
        if (chunks.empty()) {
            throw std::runtime_error("No data found in the input file.");
        }

        // Step 2: Build local dictionaries in parallel
        size_t actual_threads = std::min(thread_num, chunks.size());
        std::vector<DictMap> local_dicts(actual_threads);
        std::vector<std::thread> threads_build;

        for (size_t i = 0; i < actual_threads; ++i) {
            threads_build.emplace_back([&, i]() {
                for (size_t j = i; j < chunks.size(); j += actual_threads) {
                    build_local_dictionary(chunks[j], local_dicts[i]);
                }
            });
        }

        for (auto& t : threads_build) {
            if (t.joinable()) {
                t.join();
            }
        }

        // Step 3: Merge dictionaries
        std::mutex dict_mutex;
        std::atomic<uint32_t> current_id(0);
        merge_dictionaries(local_dicts, dictionary, dict_mutex, current_id, actual_threads);

        // Step 4: Encode data in parallel using std::async
        std::vector<std::future<std::vector<uint32_t>>> futures;
        for (size_t i = 0; i < actual_threads; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&, i]() -> std::vector<uint32_t> {
                std::vector<uint32_t> local_encoded;
                for (size_t j = i; j < chunks.size(); j += actual_threads) {
                    auto chunk_encoded = encode_chunk(chunks[j], dictionary);
                    local_encoded.insert(local_encoded.end(), chunk_encoded.begin(), chunk_encoded.end());
                }
                return local_encoded;
            }));
        }

        // Collect encoded data
        for (auto& fut : futures) {
            auto chunk_encoded = fut.get();
            encoded_data.insert(encoded_data.end(), chunk_encoded.begin(), chunk_encoded.end());
        }

        // Step 5: Build ID to indices map
        build_id_to_indices();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error during encoding: " << ex.what() << std::endl;
        throw; // Re-throw the exception after logging
    }
}

// Merge local dictionaries into global dictionary with thread_num control
void 
DictionaryEncoder::merge_dictionaries(const std::vector<DictMap>& local_dicts, DictMap& global_dict, std::mutex& dict_mutex, std::atomic<uint32_t>& current_id, size_t thread_num) {
    // Using a thread pool approach to parallelize merging
    std::vector<std::thread> threads_merge;
    for (size_t i = 0; i < thread_num; ++i) {
        threads_merge.emplace_back([&, i]() {
            for (size_t j = i; j < local_dicts.size(); j += thread_num) {
                for (const auto& [key, _] : local_dicts[j]) {
                    std::lock_guard<std::mutex> lock(dict_mutex);
                    if (global_dict.find(key) == global_dict.end()) {
                        global_dict[key] = current_id++;
                    }
                }
            }
        });
    }

    for (auto& t : threads_merge) {
        if (t.joinable()) {
            t.join();
        }
    }
}

// Serialization
void 
DictionaryEncoder::serialize(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for serialization.");
    }

    // Use a buffer to batch writes
    std::vector<char> buffer;
    buffer.reserve(4096); // Adjust buffer size as needed

    // Write dictionary size
    uint32_t dict_size = dictionary.size();
    ofs.write(reinterpret_cast<const char*>(&dict_size), sizeof(dict_size));

    // Write dictionary entries
    for (const auto& [key, id] : dictionary) {
        uint32_t key_size = key.size();
        ofs.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        ofs.write(key.data(), key_size);
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }

    // Write encoded data size
    uint32_t data_size = encoded_data.size();
    ofs.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    // Write encoded data
    ofs.write(reinterpret_cast<const char*>(encoded_data.data()), data_size * sizeof(uint32_t));
}


// Deserialization
void 
DictionaryEncoder::deserialize(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for deserialization.");
    }

    // Read dictionary size
    uint32_t dict_size;
    ifs.read(reinterpret_cast<char*>(&dict_size), sizeof(dict_size));

    // Read dictionary entries
    dictionary.clear();
    for (uint32_t i = 0; i < dict_size; ++i) {
        uint32_t key_size;
        ifs.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        ifs.read(&key[0], key_size);
        uint32_t id;
        ifs.read(reinterpret_cast<char*>(&id), sizeof(id));
        dictionary[key] = id;
    }

    // Read encoded data size
    uint32_t data_size;
    ifs.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

    // Read encoded data
    encoded_data.resize(data_size);
    ifs.read(reinterpret_cast<char*>(encoded_data.data()), data_size * sizeof(uint32_t));

    // Rebuild ID to indices map
    build_id_to_indices();
}

// Read file in chunks
std::vector<std::vector<std::string>>
DictionaryEncoder::read_file_in_chunks(const std::string& filename, size_t chunk_size_bytes) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open input file.");
    }

    std::vector<std::vector<std::string>> chunks;
    std::vector<std::string> current_chunk;
    current_chunk.reserve(100000); // Adjust based on data

    std::string line;
    size_t current_size = 0;

    while (std::getline(infile, line)) {
        current_chunk.emplace_back(std::move(line));
        current_size += line.size() + 1; // +1 for newline

        if (current_size >= chunk_size_bytes) {
            chunks.emplace_back(std::move(current_chunk));
            current_chunk = std::vector<std::string>();
            current_chunk.reserve(100000);
            current_size = 0;
        }
    }

    if (!current_chunk.empty()) {
        chunks.emplace_back(std::move(current_chunk));
    }

    return chunks;
}

// Build local dictionary
void 
DictionaryEncoder::build_local_dictionary(const std::vector<std::string>& data_chunk, DictMap& local_dict) {
    for (const auto& item : data_chunk) {
        local_dict.emplace(item, 0);
    }
}


// Implement the encode_from_memory method
void 
DictionaryEncoder::encode_from_memory(const std::vector<std::string>& raw_data, size_t thread_num) {
    // Implementation for encoding data from memory
    // This should be similar to your existing encode_from_file method,
    // but operates on the raw_data vector instead of reading from a file.
    // Ensure that this method populates the 'dictionary' and 'encoded_data' members.
    
    try {
        // Step 1: Split raw_data into chunks based on thread_num
        size_t total_entries = raw_data.size();
        size_t chunk_size = total_entries / thread_num;
        std::vector<std::vector<std::string>> chunks(thread_num);
        
        for (size_t i = 0; i < thread_num; ++i) {
            size_t start_idx = i * chunk_size;
            size_t end_idx = (i == thread_num - 1) ? total_entries : start_idx + chunk_size;
            chunks[i].assign(raw_data.begin() + start_idx, raw_data.begin() + end_idx);
        }
        
        // Step 2: Build local dictionaries in parallel
        std::vector<std::unordered_map<std::string, uint32_t>> local_dicts(thread_num);
        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < thread_num; ++i) {
            threads.emplace_back([&, i]() {
                for (const auto& str : chunks[i]) {
                    local_dicts[i][str] = 0; // Temporary value; actual ID assignment happens later
                }
            });
        }
        
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        
        // Step 3: Merge local dictionaries into a global dictionary with unique IDs
        std::mutex dict_mutex;
        uint32_t current_id = 0;
        for (size_t i = 0; i < thread_num; ++i) {
            for (const auto& [str, _] : local_dicts[i]) {
                std::lock_guard<std::mutex> lock(dict_mutex);
                if (dictionary.find(str) == dictionary.end()) {
                    dictionary[str] = current_id++;
                }
            }
        }
        
        // Step 4: Encode data in parallel
        encoded_data.resize(total_entries);
        threads.clear();
        
        for (size_t i = 0; i < thread_num; ++i) {
            threads.emplace_back([&, i]() {
                for (size_t j = i * chunk_size; j < (i == thread_num - 1 ? total_entries : (i + 1) * chunk_size); ++j) {
                    encoded_data[j] = dictionary[raw_data[j]];
                }
            });
        }
        
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        
        // Step 5: Build ID to indices map
        build_id_to_indices();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error during encoding from memory: " << ex.what() << std::endl;
        throw; // Re-throw after logging
    }
}


// Encode a single chunk
std::vector<uint32_t> 
DictionaryEncoder::encode_chunk(const std::vector<std::string>& data_chunk, const DictMap& global_dict) const {
    std::vector<uint32_t> encoded;
    encoded.reserve(data_chunk.size());

    for (const auto& item : data_chunk) {
        auto it = global_dict.find(item);
        if (it != global_dict.end()) {
            encoded.emplace_back(it->second);
        } else {
            // Handle unknown key
            encoded.emplace_back(UINT32_MAX); // Placeholder
        }
    }

    return encoded;
}


// Build ID to indices map for querying
void 
DictionaryEncoder::build_id_to_indices() {
    id_to_indices.clear();
    size_t num_threads = std::thread::hardware_concurrency();
    size_t data_size = encoded_data.size();
    size_t chunk_size = data_size / num_threads;

    std::vector<std::unordered_map<uint32_t, std::vector<size_t>>> local_id_to_indices(num_threads);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? data_size : start + chunk_size;
        threads.emplace_back([this, &local_id_to_indices, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                uint32_t id = encoded_data[j];
                local_id_to_indices[i][id].push_back(j);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Merge local_id_to_indices into id_to_indices
    for (const auto& local_map : local_id_to_indices) {
        for (const auto& [id, indices] : local_map) {
            id_to_indices[id].insert(id_to_indices[id].end(), indices.begin(), indices.end());
        }
    }
}


