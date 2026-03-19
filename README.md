# CUtrack
Multi-object tracking with CUDA


## CUTrack Project Package and Layout

pending

* [CUTRACK/](.): The parent package folder.
    * [README.md](README):
      The guide
    * [`hungarian_algo_test.cu`](hungarian_algo_test.cu) 
    This is the test function for comparing different types of CUDA kernel for hungarian algorithm
    * [`helper.h`](helper.h) 
    This file is the header file for helper function.
    * [`sort_lib.h`](sort_lib.h) 
    This file is the header file for all CUDA accelerated functions for the SORT algorithm.
    * [`hungarian_lib.cu`](hungarian_lib.cu)
    This is the main library of all cuda kernels
    * [`hungarian_cpu_vectorized.h`](hungarian_cpu_vectorized.h)
    This the AVX accelerated CPU version of hungarian algorithm for benckmarking.
