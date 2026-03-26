#ifndef sort_lib_h
#define sort_lib_h

#include <cuda_runtime.h>



// A wrapper function that calls the GPU kernel
void reductionStreamMemory(float* input, int totalsize, int blocksize, int width, int height);

void reductionmappedmem(float* input, int totalsize, int blocksize, int width, int height);

void reductionglobalmem(float* input, int totalsize, int blocksize, int width, int height);

void reductionNoTransposeStreamMemory(float* input, int totalsize, int blocksize, int width, int height);

void reductionNotranspose(float* input, int totalsize, int blocksize, int width, int height);

// If you want to use a __device__ function across files, use 'extern'
// Note: This requires -rdc=true during compilation
// extern __device__ float GPU_Helper_Math(float x);




#endif