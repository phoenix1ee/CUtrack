#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cinttypes>

#include <immintrin.h>
#include <algorithm>
#include <float.h>

inline void row_min_subtraction_avx(float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float* row_ptr = data + (i * cols);
        
        // --- Pass 1: Find Row Minimum ---
        __m256 vmin = _mm256_set1_ps(FLT_MAX); 
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 vdata = _mm256_loadu_ps(&row_ptr[j]);
            vmin = _mm256_min_ps(vmin, vdata);
        }

        // Fold the 8-way vector min down to a single scalar
        float temp[8];
        _mm256_storeu_ps(temp, vmin);
        float row_min = FLT_MAX;
        for (int k = 0; k < 8; ++k) row_min = min(row_min, temp[k]);

        // Cleanup for remaining elements in Pass 1
        for (; j < cols; ++j) {
            row_min = min(row_min, row_ptr[j]);
        }

        // --- Pass 2: Subtract Minimum ---
        __m256 vrow_min = _mm256_set1_ps(row_min); // Broadcast min to all 8 slots
        j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 vdata = _mm256_loadu_ps(&row_ptr[j]);
            __m256 vresult = _mm256_sub_ps(vdata, vrow_min);
            _mm256_storeu_ps(&row_ptr[j], vresult);
        }

        // Cleanup for remaining elements in Pass 2
        for (; j < cols; ++j) {
            row_ptr[j] -= row_min;
        }
    }
}

inline void col_min_subtraction_avx(float* data, int rows, int cols) {
    int j = 0;

    // Process 8 columns at a time
    for (; j <= cols - 8; j += 8) {
        // --- Pass 1: Find Min for these 8 columns ---
        __m256 vmin = _mm256_set1_ps(FLT_MAX);
        for (int i = 0; i < rows; ++i) {
            __m256 vdata = _mm256_loadu_ps(&data[i * cols + j]);
            vmin = _mm256_min_ps(vmin, vdata);
        }

        // --- Pass 2: Subtract Min for these 8 columns ---
        for (int i = 0; i < rows; ++i) {
            __m256 vdata = _mm256_loadu_ps(&data[i * cols + j]);
            __m256 vresult = _mm256_sub_ps(vdata, vmin);
            _mm256_storeu_ps(&data[i * cols + j], vresult);
        }
    }

    // --- Cleanup: Handle remaining columns (if cols % 8 != 0) ---
    for (; j < cols; ++j) {
        float col_min = FLT_MAX;
        for (int i = 0; i < rows; ++i) {
            col_min = min(col_min, data[i * cols + j]);
        }
        for (int i = 0; i < rows; ++i) {
            data[i * cols + j] -= col_min;
        }
    }
}


inline void reduction_avx(float* input, int width, int height){

    row_min_subtraction_avx(input,height,width);
    col_min_subtraction_avx(input,height,width);

}
