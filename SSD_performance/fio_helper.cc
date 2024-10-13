#include "fio_helper.hh"
#include <iostream>
#include <cstdlib>  // for system()
#include <sstream>

std::string generate_fio_command(const FioTestParams& params, const std::string& test_file, int idx) {
    std::ostringstream cmd;

    //cmd << "Command applied: " << params.access_size_kb << "KB, " << params.read_ratio << " ratio on read, " << params.queue_depth << " depth.";

    // Set base command with data access size and queue depth
    cmd << ".\\fio\\fio.exe --name=test --rw=randrw --bs=" << params.access_size_kb << "K --ioengine=windowsaio ";
    cmd << "--iodepth=" << params.queue_depth << " --direct=1 --size=1G --numjobs=1 ";
    
    // Set the read vs write ratio
    if (params.read_ratio == 1.0) {
        cmd << "--rw=read ";
    } else if (params.read_ratio == 0.0) {
        cmd << "--rw=write ";
    } else {
        cmd << "--rwmixread=" << params.read_ratio * 100 << " --rwmixwrite=" << (1 - params.read_ratio) * 100 << " ";
    }

    // Specify the test file to use
    cmd << "--filename=" << test_file << " --output=.\\data\\fio_output" << idx << ".txt";

    return cmd.str();
}

void run_fio_test(const FioTestParams& params, const std::string& test_file, int idx) {
    std::string cmd = generate_fio_command(params, test_file, idx);

    // Print the command to be executed for reference
    std::cout << "Running FIO test with command: " << cmd << std::endl;

    // Run the FIO command
    int result = system(cmd.c_str());

    if (result == 0) {
        std::cout << "FIO test completed successfully!" << std::endl;
    } else {
        std::cerr << "FIO test failed!" << std::endl;
    }
}
