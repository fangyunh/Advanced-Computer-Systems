#include "fio_helper.hh"
#include <iostream>

int main() {
    std::string test_file = ".\\fio_testfile.dat";  // Modify to your partition

    // Define different combinations of parameters
    std::vector<FioTestParams> test_cases = {
        {4, 1.0, 64},   
        {4, 0.0, 64},   
        {4, 0.5, 64},
        {4, 0.7, 64},
        {4, 1.0, 512},   
        {4, 0.0, 512},   
        {4, 0.5, 512},
        {4, 0.7, 512},
        {4, 1.0, 1024},   
        {4, 0.0, 1024},   
        {4, 0.5, 1024},
        {4, 0.7, 1024},

        {32, 1.0, 64},   
        {32, 0.0, 64},   
        {32, 0.5, 64},
        {32, 0.7, 64},
        {32, 1.0, 512},   
        {32, 0.0, 512},   
        {32, 0.5, 512},
        {32, 0.7, 512},
        {32, 1.0, 1024},   
        {32, 0.0, 1024},   
        {32, 0.5, 1024},
        {32, 0.7, 1024},


        {128, 1.0, 64},   
        {128, 0.0, 64},   
        {128, 0.5, 64},
        {128, 0.7, 64},
        {128, 1.0, 512},   
        {128, 0.0, 512},   
        {128, 0.5, 512},
        {128, 0.7, 512},
        {128, 1.0, 1024},   
        {128, 0.0, 1024},   
        {128, 0.5, 1024},
        {128, 0.7, 1024},

        {512, 1.0, 64},   
        {512, 0.0, 64},   
        {512, 0.5, 64},
        {512, 0.7, 64},
        {512, 1.0, 512},   
        {512, 0.0, 512},   
        {512, 0.5, 512},
        {512, 0.7, 512},
        {512, 1.0, 1024},   
        {512, 0.0, 1024},   
        {512, 0.5, 1024},
        {512, 0.7, 1024}
        
    };
    int i = 0;
    for (const auto& params : test_cases) {
        std::cout << "Testing access size " << params.access_size_kb << "KB, ";
        std::cout << "Read ratio: " << params.read_ratio * 100 << "%, ";
        std::cout << "Queue depth: " << params.queue_depth << std::endl;
        // Run the FIO test with the given parameters
        run_fio_test(params, test_file, i);
        i++;
    }

    return 0;
}
