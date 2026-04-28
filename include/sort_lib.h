#ifndef sort_lib_h
#define sort_lib_h

#include <cuda_runtime.h>

#include <string>

//tracker struct wrapper
typedef struct tracker{
    int Max_Tracks;int Max_detection;   //N and M
    int m;int n;    

    char* d_base; //base address for all device memory

    int* d_track_id;    //track IDs   N
    float * d_state_predicted;    //state variables   n*N
    float * d_state_updated;    //state variables   n*N
    float * d_F;                  //transition matrix n*n, row major storage
    float * d_Q;            //process noise   n*n  
    float* d_Pcov;         //state covariance matrix   n*n*N
    float* d_Pcov_predict; //predicted state covariance matrix   n*n*N

    int* d_age;         //age of each tracks    N
    int* d_hit_streak;  //consecutive hit for each track   N
    float* d_Z;         // measurement/dection buffer, m*M,   row major
    float* d_y;          // innovation buffer, m*N,   row major
    float* d_S;float* d_K;float* d_H;float* d_R; //matrix for kalman gain
    //     N*m*m     N*n*m       m*n        m*m
    //              col major   col major    col major
    float** d_each_K;float** d_each_S;int* d_info;    //buffer space for cublas and cusolver
    
    float* d_IOU;        //buffer for IOU calculation, N*M
    float* d_bid;   //buffer for bid value matrix, N*M
    int* d_bid_target;   //buffer for bid target value matrix, N
    float* d_price;     //buffer for price of each detection, M
    int* d_match_detections;     //buffer for assignment of detections to tracks, M
    int* d_match_track;     //buffer for assigned detections for each tracks, N
    

    //constructor
    tracker(int MT=2000,int MD=2000, int measure=4, int state=7):
        Max_Tracks(MT),Max_detection(MD), m(measure),n(state){}

    //allocate device memory for member matrix
    void allocateOnDevice(){
        //allocate 1 big block and partition for different arrays
        //all types are 4byte so no alignment problem
        size_t total = sizeof(int)*Max_Tracks
                        +sizeof(float)*Max_Tracks*n
                        +sizeof(float)*Max_Tracks*n
                        +sizeof(float)*n*n
                        +sizeof(float)*n*n
                        +sizeof(float)*Max_Tracks*n*n
                        +sizeof(float)*Max_Tracks*n*n
                        +sizeof(int)*Max_Tracks
                        +sizeof(int)*Max_Tracks
                        +sizeof(float)*Max_detection*m
                        +sizeof(float)*Max_Tracks*m
                        +sizeof(float)*m*m*Max_Tracks
                        +sizeof(float)*n*m*Max_Tracks
                        +sizeof(float)*n*m
                        +sizeof(float)*m*m
                        +sizeof(int)*Max_Tracks
                        +sizeof(float)*Max_Tracks*Max_detection
                        +sizeof(float)*Max_Tracks*Max_detection
                        +sizeof(int)*Max_Tracks
                        +sizeof(float)*Max_detection
                        +sizeof(int)*Max_detection
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

        d_Q=(float*)(d_base+offset);
        offset+=sizeof(float)*n*n;            //n*n        

        d_Pcov=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n*n;  //n*n*Max_Tracks

        d_Pcov_predict=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*n*n;  //n*n*Max_Tracks

        d_age=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;     //1*Max_Tracks

        d_hit_streak=(int*)(d_base+offset);
        offset+=sizeof(int)*Max_Tracks;       //1*Max_Tracks

        d_Z=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_detection*m; //Max_detection*m

        d_y=(float*)(d_base+offset);
        offset+=sizeof(float)*Max_Tracks*m; //Max_detection*m

        d_S=(float*)(d_base+offset);
        offset+=sizeof(float)*m*m*Max_Tracks;  //m*m*Max_Tracks

        d_K=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m*Max_Tracks;  //n*m*Max_Tracks

        d_H=(float*)(d_base+offset);
        offset+=sizeof(float)*n*m;             //n*m

        d_R=(float*)(d_base+offset);
        offset+=sizeof(float)*m*m;             //m*m

        d_info=(int*)(d_base+offset);         //1*Max_Tracks
        offset+=sizeof(int)*Max_Tracks;

        d_price=(float*)(d_base+offset);         //Max_Tracks
        offset+=sizeof(int)*Max_Tracks;

        d_bid_target=(int*)(d_base+offset);         //Max_tracks
        offset+=sizeof(int)*Max_Tracks;

        d_match_detections=(int*)(d_base+offset);         //Max_detection
        offset+=sizeof(int)*Max_detection;

        d_match_track=(int*)(d_base+offset);         //Max_tracks
        offset+=sizeof(int)*Max_Tracks;

        d_bid =(float*)(d_base+offset);         //Max_Detection*Max_Tracks
        offset+=sizeof(float)*Max_detection*Max_Tracks;

        d_IOU=(float*)(d_base+offset);         //Max_Detection*Max_Tracks


        //the array of pointers for cusolver/cublas
        cudaMalloc((void**)&d_each_K, sizeof(float*) * Max_Tracks);
        cudaMalloc((void**)&d_each_S, sizeof(float*) * Max_Tracks);
        float** h_each_K = (float**)malloc(sizeof(float*)*Max_Tracks);
        float** h_each_S = (float**)malloc(sizeof(float*)*Max_Tracks);
        for(int i=0;i<Max_Tracks;i++){
            h_each_K[i]=d_K+i*n*m;
            h_each_S[i]=d_S+i*m*m;
        }
        cudaMemcpy(d_each_K,h_each_K,sizeof(float*)*Max_Tracks,cudaMemcpyHostToDevice);
        cudaMemcpy(d_each_S,h_each_S,sizeof(float*)*Max_Tracks,cudaMemcpyHostToDevice);
        free(h_each_K);
        free(h_each_S);

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
//matrix addition-1 to many-batch mode
__global__ void MMAdd1toMany(float* batchedA, float* singleB, int strideA, int row, int col, int batchCount);

//wrapper function for input of data
struct ImageData load_jpeg_bgr_hwc_to_host(const std::string& path);
struct ImageData load_jpeg_rgb_hwc_to_host(const std::string& path);
void free_jpeg_from_host(ImageData image);

//wrapper function and helper function-image processing for onnx runtime
__device__ float boxIOU(float acx,float acy,float aw,float ah,
                        float bcx,float bcy,float bw,float bh);
void frame_preprocess(uint8_t* d_frame_in,float* d_frame_out,int h_in, int w_in, int h_out, int w_out);

//wrapper function-post processing after detections
void NMS(float* d_raw_detections, int* d_raw_class_id, int Num_raw_detection, int height_raw_detection, 
        float* d_buffer_detections, int* d_buffer_class_id, int*d_detection_count);
void copyToTracker(float* d_detector_output, float* d_Z,int Num_raw_detection,int detection_count);

//wrapper function for state udate
void set_first_state(tracker &tracker, int num_current_tracks, int num_added_tracks);
void set_first_Pcov(tracker &tracker, int current_tracks, int num_added_tracks);
void set_first_age_hit(tracker &tracker, int num_current_tracks, int num_added_tracks);
void set_single_F(tracker &tracker);
void set_single_R(tracker &tracker);
void set_single_Q(tracker &tracker);
void set_single_H(tracker &tracker);
void make_state_prediction(tracker &tracker, int num_current_tracks);
void make_cov_prediction(tracker &tracker, int num_current_tracks);
void update_states_Kalman(tracker &tracker, int num_current_tracks);

//wrapper function for IOU calculation
void tracker_compute_IOU(tracker &tracker, int activetrack, int activedetection);

//wrapper function that calls GPU kernels / library for Kalman filter
void cublasTranspose_simple(float* d_in, float* d_out, int o_row, int o_col);
void cublasadd_simple(float* d_A, float* d_B, int n);
void cublasmmulti(float* d_A, float* d_B, float* d_C, bool A_T, bool B_T, int m, int n, int k);
void kalman_gain_single(float* d_K, float*d_P,float*d_H,float*d_R,int m, int n);
void predict_positions(float* d_F, float* d_x, int num_objects);
void kalman_gain_batch(bool*inactive, float* d_S, float*d_PHT, float*d_P,float*d_H,float*d_R,int totaltracks, int m, int n);
void tracker_kalman_gain(tracker* trackerA, int totaltracks);

//wrapper function that calls the GPU kernel for hungarian algorithm(different design)
void hungarian_assignment(tracker &tracker,int width_detections, int height_tracks);
void transposeArray(float *d_a,float *d_b,int matrixwidth, int matrixheight);
void reductionStreamMemory(float* input, int totalsize, int blocksize, int width, int height);
void reductionmappedmem(float* input, int totalsize, int blocksize, int width, int height);
void reductionglobalmem(float* input, int totalsize, int blocksize, int width, int height);
void reductionNoTransposeStreamMemory(float* input, int totalsize, int blocksize, int width, int height);
void reductionNotranspose(float* input, int totalsize, int blocksize, int width, int height);

// If you want to use a __device__ function across files, use 'extern'
// Note: This requires -rdc=true during compilation
// extern __device__ float GPU_Helper_Math(float x);




#endif