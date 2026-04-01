#ifndef sort_lib_h
#define sort_lib_h

#include <cuda_runtime.h>

//tracker struct wrapper
typedef struct tracker{
    int Max_Tracks;int Max_detection;   //N and M
    int m;int n;    

    char* d_base; //base address for memory

    int* d_track_id;    //track IDs   N
    float * d_state_predicted;    //state variables   n*N
    float * d_state_updated;    //state variables   n*N
    float * d_F;                  //transition matrix n*n
    float* d_Pcov;       //state covariance matrix   n*n*N
    int* d_age;         //age of each tracks    N
    int* d_hit_streak;  //consecutive hit for each track   N
    int* d_activetracks; // number of active tracks    1
    float* d_Z;         // measurement/dection buffer, m*M
    int* d_currentdetections; // number of detection get    1
    float* d_S;float* d_K;float* d_H;float* d_R; //matrix for kalman gain
    //     N*m*m     N*n*m       m*n        m*m
    float** d_each_K;float** d_each_S;int* d_info;    //buffer space for cublas and cusolver
    
    //constructor
    tracker(int MT=2000,int MD=2000, int measure=4, int state=7):
        Max_Tracks(MT),Max_detection(MD), m(measure),n(state) {}

    //allocate device memory for member matrix
    void allocateOnDevice(){
        //allocate 1 big block and partition for different arrays
        //all types are 4byte so no alignment problem
        size_t total = sizeof(int)*Max_Tracks
                        +sizeof(float)*Max_Tracks*n
                        +sizeof(float)*Max_Tracks*n
                        +sizeof(float)*n*n
                        +sizeof(float)*Max_Tracks*n*n
                        +sizeof(int)*Max_Tracks
                        +sizeof(int)*Max_Tracks
                        +sizeof(int)
                        +sizeof(float)*Max_detection*m
                        +sizeof(int)
                        +sizeof(float)*m*m*Max_Tracks
                        +sizeof(float)*n*m*Max_Tracks
                        +sizeof(float)*n*m
                        +sizeof(float)*m*m
                        +sizeof(int)*Max_Tracks;

        cudaMalloc((void**)&d_base,total);
        size_t offset = 0;

        d_track_id=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;       //1*Max_Tracks

        d_state_predicted=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n;   //n*Max_Tracks

        d_state_updated=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n;   //n*Max_Tracks        

        d_F=(float*)(d_base+offset);
        offset+=sizeof(float)*n*n;            //n*n        

        d_Pcov=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n*n;  //n*n*Max_Tracks

        d_age=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;     //1*Max_Tracks

        d_hit_streak=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;       //1*Max_Tracks

        d_activetracks=(int*)(d_base+offset);
        offset+=sizeof(int);                  //1

        d_Z=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_detection*m; //Max_detection*m

        d_currentdetections =(int*)(d_base+offset);
        offset+=sizeof(int);                   //1

        d_S=(float*)(d_base+offset);
        offset+=sizeof(float)*m*m*Max_Tracks;  //m*m*Max_Tracks

        d_K=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m*Max_Tracks;  //n*m*Max_Tracks

        d_H=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m;             //n*m

        d_R=(float*)(d_base+offset);
        offset+=sizeof(float)*m*m;             //m*m

        d_info=(int*)(d_base+offset);         //1*Max_Tracks

        cudaMalloc((void**)&d_each_K, sizeof(float*) * Max_Tracks);
        cudaMalloc((void**)&d_each_S, sizeof(float*) * Max_Tracks);
    }
    //free the device tracker object matrix
    void freeOnDevice(){
        cudaFree(d_base);   
        cudaFree(d_each_K);
        cudaFree(d_each_S);
    }

}tracker;

//print device matrix on host
void print_device_matrix_colmajor_from_host(float*d_input,int cols,int rows);
__global__ void print_device_matrix_kernel(float*d_input,int cols,int rows);

//custom kernels
//initialize the individual matrix array pointer
__global__ void KSaddrInitialize(tracker *d_tracker);
//matrix addition-1 to many-batch mode
__global__ void MMAdd1toMany(float* batchedA, float* singleB, int row, int col, int batchCount);

//IOU calculation
__global__ void computeIOUmatrix(float* d_predictedstate, float* d_detectbox, float* d_IOUmatrix, int Ntracks, int Mdetection, int image_w, int image_h);

//wrapper function that calls GPU kernels / library for Kalman filter
void cublasTranspose_simple(float* d_in, float* d_out, int o_row, int o_col);
void cublasadd_simple(float* d_A, float* d_B, int n);
void cublasmmulti(float* d_A, float* d_B, float* d_C, bool A_T, bool B_T, int m, int n, int k);
void kalman_gain_single(float* d_K, float*d_P,float*d_H,float*d_R,int m, int n);
void predict_positions(float* d_F, float* d_x, int num_objects);
void kalman_gain_batch(bool*inactive, float* d_S, float*d_PHT, float*d_P,float*d_H,float*d_R,int totaltracks, int m, int n);
void tracker_kalman_gain(tracker* trackerA, int totaltracks);

//wrapper function that calls the GPU kernel for hungarian algorithm(different design)
void reductionStreamMemory(float* input, int totalsize, int blocksize, int width, int height);
void reductionmappedmem(float* input, int totalsize, int blocksize, int width, int height);
void reductionglobalmem(float* input, int totalsize, int blocksize, int width, int height);
void reductionNoTransposeStreamMemory(float* input, int totalsize, int blocksize, int width, int height);
void reductionNotranspose(float* input, int totalsize, int blocksize, int width, int height);

// If you want to use a __device__ function across files, use 'extern'
// Note: This requires -rdc=true during compilation
// extern __device__ float GPU_Helper_Math(float x);




#endif