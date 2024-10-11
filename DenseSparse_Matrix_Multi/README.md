# Dense/Sparse Matrix-Matrix Multiplication

**Compilation Command**: 

```bash
g++ -o matrix_mul matrix_main.cc matrix_multi.cc -fopenmp -mavx2 -march=native -std=c++17 -O3 -g
```



**Single Run Command (options in () are optional, options in [] are parameter):**

```bash
./matrix_mul -n [size] -t [dd|ds|ss] (-m [thread_num]) (-s) (-o)
```

-n: size of matrix

-t: **dd** for dense * dense, **ds** for dense * sparse, **ss** for sparse * sparse

-m: multi-threading enable with thread number

-s: SIMD enable

-o: matrix compression optimization enable



**test_script.py**: Run all experiments at once. Results are written in *test_results.txt*

```bash
python ./test_script.py
```



**matrix_main.cc**: The main logic to run the maltiplication.

**matrix_multi.hh**: Define the utilities functions and class.

**matrix_multi.cc**: Implementation of input handling,  all multiplications, and compression.

test_results(latency).txt: records latency data in the test



PS: For 10,000 * 10,000 matrix, the test script fails to get results. So we manually did it and recorded in the test_results.txt