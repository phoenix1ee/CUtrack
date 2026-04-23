// Create and return a pointer to an array of size rows and cols
// populate with random value

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cinttypes>

typedef struct ImageData {
    uint8_t* data;   // BGR HWC buffer
    size_t size;     // total bytes
    int width;
    int height;
}ImageData;

//print matrix on device kernel
__device__ inline void printmatrix_colmajor_ondevice(float*d_input,int cols,int rows){
    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
				printf("%6.3f ",d_input[j*rows+i]);
		}
		printf("\n");
	}
}

//function to calculate offset with alignment
inline size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

inline float* create_2d_array(int cols,int rows) {
    float* m = (float*)malloc(rows * cols * sizeof(float));
	srand(time(NULL));
    for (int i = 0; i < (rows * cols); i++) {
		// Initialize with random number between 1-999
        m[i] = rand() % 999+1;		
    }
    return m;
}

inline void checkmatrix(float*input,float*input2,int size){
    for(int i=0;i<size;i++){
        if (input[i]!=input2[i]){
            printf("result does not match. Error detected");
            return;
        }
    } 
}

inline void printmatrix(float*input,int cols,int rows){
    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
		
				printf("%6.2f ",input[i*cols+j]);
		}
		printf("\n");
	}
}

inline void printmatrix_colmajor(float*input,int cols,int rows){
    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
				printf("%6.3f ",input[j*rows+i]);
		}
		printf("\n");
	}
}

inline void write2DArrayToFile(const float* data, int rows, int cols, const char* filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            file << data[i * cols + j];

            if (j < cols - 1)
                file << ",";  // comma-separated
        }
        file << "\n"; // new row
    }

    file.close();
}

inline void write2DArrayToFileInt(const int* data, int rows, int cols, const char* filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            file << data[i * cols + j];

            if (j < cols - 1)
                file << ",";  // comma-separated
        }
        file << "\n"; // new row
    }

    file.close();
}

#include <cuda_runtime.h>

#include <time.h>
// get time stamp for cudaevent
inline __host__ cudaEvent_t cudagettime(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// For PRIu64
#include <cinttypes>
//multi platform time function for host
#if defined(_WIN32)
#include <windows.h>
// Windows replacement for struct timespec using QPC
inline void get_monotonic_timespec(struct timespec *ts) {
    static LARGE_INTEGER freq;
    static int initialized = 0;

    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    // Convert QPC ticks → nanoseconds
    uint64_t ns = (uint64_t)(counter.QuadPart * 1000000000ull / freq.QuadPart);

    ts->tv_sec  = ns / 1000000000ull;
    ts->tv_nsec = ns % 1000000000ull;
}
inline void getstarttime(struct timespec *ts){
	get_monotonic_timespec(ts);
}
inline uint64_t get_lapsed(struct timespec start) {
    struct timespec end;
    get_monotonic_timespec(&end);
    uint64_t diff =
        (uint64_t)(end.tv_sec - start.tv_sec) * 1000000000ull +
        (uint64_t)(end.tv_nsec - start.tv_nsec);

    return diff;
}

#else
//linux version
inline void getstarttime(struct timespec *ts){
	clock_gettime(CLOCK_MONOTONIC, ts);
}
inline uint64_t get_lapsed(struct timespec start) {
    struct timespec end;
	// Get current time from the monotonic clock
    clock_gettime(CLOCK_MONOTONIC, &end);
	// Calculate difference: (seconds * 1e9) + nanoseconds
    uint64_t diff =(uint64_t)(end.tv_sec - start.tv_sec) * 1000000000ull
					+(uint64_t)(end.tv_nsec - start.tv_nsec);
    return diff;
}
#endif
