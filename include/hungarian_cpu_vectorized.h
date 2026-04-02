#ifndef HUNGARIAN_CPU_VECTORIZED_H
#define HUNGARIAN_CPU_VECTORIZED_H

// Declarations that both NVCC and G++ can see
void reduction_avx(float* input, int width, int height);

#endif