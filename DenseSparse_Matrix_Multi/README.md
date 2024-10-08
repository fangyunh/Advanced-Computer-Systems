# Dense/Sparse Matrix-Matrix Multiplication

## Introduction

The objective of this design project is to implement a C/C++ module that carries out high-speed dense/sparse matrix-matrix multiplication by explicitly utilizing (i) multiple threads, (ii) x86 SIMD instructions, and/or (iii) techniques to minimize cache miss rate via restructuring data access patterns or data compression (as discussed in class). Matrix-matrix multiplication is one of the most important data processing kernels in numerous real-life applications, e.g., machine learning, computer vision, signal processing, and scientific computing. This project aims to gain hands-on experience of multi-thread programming, SIMD programming, and cache access optimization. It will develop a deeper understanding of the importance of exploiting task/data-level parallelism and minimizing cache miss rate.

## Implementation	

The implementation should be able to support configurable matrix size that can be much larger than the on-chip cache capacity. Moreover, the implementation should allow users to individually turn on/off the three  optimization techniques (i.e., multi-threading, SIMD, and cache miss minimization) and configure the thread  number so that users could easily observe the effect of any combination of these three optimization techniques.  Other than the source code, the Github site contains  

(1) Readme that clearly explains the structure/installation/usage of the code 

(2) Experimental results that show the performance of your code under different matrix size (at least including  1,000x1,000 and 10,000x10,000) and different matrix sparsity (at least including 1% and 0.1%) 

(3) Present and compare the performance of (i) native implementation of matrix-matrix multiplication without  any optimization, (ii) using multi-threading only, (iii) using SMID only, (iv) using cache miss optimization only,  (v) using all the three techniques together 

(4) Present the performance of (1) dense-dense matrix multiplication, (2) dense-sparse matrix multiplication,  and (3) sparse-sparse matrix multiplication 

(5) Thorough analysis and conclusion (include discussions under what matrix sparsity we would like to enable matrix compression)