#ifndef FIO_HELPER_HH
#define FIO_HELPER_HH

#include <string>
#include <vector>

struct FioTestParams {
    int access_size_kb;    // 4KB, 16KB, etc.
    float read_ratio;      // 1.0 for read-only, 0.5 for 50% read/write
    int queue_depth;       // Queue depth (0 to 1024)
};

// Function to create a test command for FIO
std::string generate_fio_command(const FioTestParams& params, const std::string& test_file, int idx);

// Function to run the FIO test and capture output
void run_fio_test(const FioTestParams& params, const std::string& test_file, int idx);

#endif
