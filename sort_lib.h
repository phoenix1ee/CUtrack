#ifndef sort_lib_h
#define sort_lib_h

#include <cuda_runtime.h>

//tracker struct wrapper
typedef struct tracker{
    int Max_Tracks;int Max_detection;
    int m;int n;    
    int* d_track_id;    //track IDs
    float * d_state;    //state variables
    float*d_Pcov;       //state covariance matrix
    int* d_age;         //age of each tracks
    int* d_hit_streak;  //consecutive hit for each track
    int* d_activetracks; // number of active tracks
    float* d_Z;         // measurement/dection buffer, m*N
    int d_currentdetections; // number of detection get
    float* d_K;float* d_H;float* d_R; //matrix for kalman gain

    //constructor
    tracker(int MT=2000,int MD=2000, int measure=4, int state=7):
        Max_Tracks(MT),Max_detection(MD), m(measure),n(state) {}

    //allocate device memory for member matrix
    void allocateOnDevice(){
        cudaMalloc((void**)&d_track_id,sizeof(int)*Max_Tracks);
        cudaMalloc((void**)&d_state,sizeof(float)*Max_Tracks*n);
        cudaMalloc((void**)&d_Pcov,sizeof(float)*Max_Tracks*n*n);
        cudaMalloc((void**)&d_age,sizeof(int)*Max_Tracks);
        cudaMalloc((void**)&d_hit_streak,sizeof(int)*Max_Tracks);
        cudaMalloc((void**)&d_activetracks,sizeof(int));
        cudaMalloc((void**)&d_Z,sizeof(float)*Max_detection*m);
        cudaMalloc((void**)&d_K,sizeof(float)*n*m*Max_Tracks);
        cudaMalloc((void**)&d_H,sizeof(float)*n*m);
        cudaMalloc((void**)&d_R,sizeof(float)*m*m);
    }
    //free the device tracker object matrix
    void freeOnDevice(){
        cudaFree(d_track_id);
        cudaFree(d_state);
        cudaFree(d_Pcov);
        cudaFree(d_age);
        cudaFree(d_hit_streak);
        cudaFree(d_activetracks);
        cudaFree(d_Z);
        cudaFree(d_K);
        cudaFree(d_H);
        cudaFree(d_R);
        
    }

}tracker;



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