// encoder.hh
#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>

class DictionaryEncoder {
public:
    using DictMap = std::unordered_map<std::string, uint32_t>;

    // Encode function: Reads, builds dictionary, and encodes data
    void encode(const std::string& filename, size_t chunk_size_bytes = 100 * 1024 * 1024, size_t thread_num = std::thread::hardware_concurrency());

    // Serialization
    void serialize(const std::string& filename) const;

    // Deserialization
    void deserialize(const std::string& filename);

    // encoder.hh
    void encode_from_memory(const std::vector<std::string>& raw_data, size_t thread_num = std::thread::hardware_concurrency());

    // Accessors
    const DictMap& get_dictionary() const { return dictionary; };
    const std::vector<uint32_t>& get_encoded_data() const { return encoded_data; };

private:
    DictMap dictionary;
    std::vector<uint32_t> encoded_data;
    std::unordered_map<uint32_t, std::vector<size_t>> id_to_indices;

    // Read file in chunks
    std::vector<std::vector<std::string>> read_file_in_chunks(const std::string& filename, size_t chunk_size_bytes);

    // Build local dictionary
    void build_local_dictionary(const std::vector<std::string>& data_chunk, DictMap& local_dict);

    // Merge local dictionaries into global dictionary
    void merge_dictionaries(const std::vector<DictMap>& local_dicts, DictMap& global_dict, std::mutex& dict_mutex, std::atomic<uint32_t>& current_id, size_t thread_num);

    // Encode a single chunk
    std::vector<uint32_t> encode_chunk(const std::vector<std::string>& data_chunk, const DictMap& global_dict) const;

    // Build ID to indices map
    void build_id_to_indices();
};
