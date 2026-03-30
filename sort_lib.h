#ifndef sort_lib_h
#define sort_lib_h

#include <cuda_runtime.h>

//tracker struct wrapper
typedef struct tracker{
    int Max_Tracks;int Max_detection;   //N and M
    int m;int n;    

    char* d_base; //base address for memory

    int* d_track_id;    //track IDs   N
    float * d_state;    //state variables   n*N
    float*d_Pcov;       //state covariance matrix   n*n*N
    int* d_age;         //age of each tracks    N
    int* d_hit_streak;  //consecutive hit for each track   N
    int* d_activetracks; // number of active tracks    1
    float* d_Z;         // measurement/dection buffer, m*M
    int* d_currentdetections; // number of detection get    1
    float* d_S;float* d_K;float* d_H;float* d_R; //matrix for kalman gain
    //     N*m*m     N*n*m       m*n        m*m

    //constructor
    tracker(int MT=2000,int MD=2000, int measure=4, int state=7):
        Max_Tracks(MT),Max_detection(MD), m(measure),n(state) {}

    //allocate device memory for member matrix
    void allocateOnDevice(){
        //allocate 1 big block and partition for different arrays
        //all types are 4byte so no alignment problem
        size_t total = (3+n+n*n+n*m+m*m)*Max_Tracks
                    +2
                    +Max_detection*m
                    +n*m
                    +m*m;

        cudaMalloc((void**)&d_base,total);
        size_t offset = 0;

        d_track_id=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;

        d_state=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n;

        d_Pcov=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n*n;

        d_age=(int*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks;

        d_hit_streak=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;

        d_activetracks=(int*)(d_base+offset);
        offset+=sizeof(int);

        d_Z=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_detection*m;

        d_currentdetections =(int*)(d_base+offset);
        offset+=sizeof(int);

        d_S=(float*)(d_base+offset);
        offset+=sizeof(float)*m*m*Max_Tracks;

        d_K=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m*Max_Tracks;

        d_H=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m;

        d_R=(float*)(d_base+offset);
    }
    //free the device tracker object matrix
    void freeOnDevice(){
        cudaFree(d_base);        
    }

}tracker;

//wrapper function that calls GPU kernels / library for Kalman filter



// A wrapper function that calls the GPU kernel for hungarian algorithm
void reductionStreamMemory(float* input, int totalsize, int blocksize, int width, int height);

void reductionmappedmem(float* input, int totalsize, int blocksize, int width, int height);

void reductionglobalmem(float* input, int totalsize, int blocksize, int width, int height);

void reductionNoTransposeStreamMemory(float* input, int totalsize, int blocksize, int width, int height);

void reductionNotranspose(float* input, int totalsize, int blocksize, int width, int height);

// If you want to use a __device__ function across files, use 'extern'
// Note: This requires -rdc=true during compilation
// extern __device__ float GPU_Helper_Math(float x);




#endif