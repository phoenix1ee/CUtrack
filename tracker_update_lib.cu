#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include "include/sort_lib.h"

__global__ void set_first_state_kernel(float* source, float* dest, int N, int start_col, int count){
    //source: 5*N row major, each detection sub element in contiguous
    //[cx,cy,s,r]
    //dest: n*N row major, each state factor in contiguous
    //[cx,cy,s,r,x.,y.,s.]
    //1 thread for each states
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>count){return;}
    for(int i = tid;i<count;i+=totalthreads){

        int col = start_col+i;
        if (col>=N)return;
        
        dest[col]=source[col];   //cx
        dest[col+N]=source[col+N];   //cy
        dest[col+2*N]=source[col+2*N];   //s
        dest[col+3*N]=source[col+3*N];   //r
        dest[col+4*N]=1e-3;   //x.
        dest[col+5*N]=1e-3;   //y.
        dest[col+6*N]=1e-3;   //s.        
    }
}

__global__ void set_first_pcov_kernel(float* dest, int N, int start_matrix_ID, int count, int matrix_dim){
    //dest: n*n*N row major, each state factor in contiguous, n=matrix_dim
    //1 thread 1 element
    //output: dest, d_Pcov at tracker, n*n*M, column major
    int totalthreads = gridDim.x*blockDim.x;
    int global_ID = blockIdx.x*blockDim.x+threadIdx.x;
    int matrixsize = matrix_dim*matrix_dim;
    __shared__ float defaultvalue[7];
    if(blockIdx.x==0 && threadIdx.x==0){
        defaultvalue[0]=1.0;
        defaultvalue[1]=1.0;
        defaultvalue[2]=1.0;
        defaultvalue[3]=0.01;
        defaultvalue[4]=10.0;
        defaultvalue[5]=10.0;
        defaultvalue[6]=1.0;
    }
    for(int i = global_ID;i<count*matrixsize;i+=totalthreads){
        int matrixID = i/matrixsize;
        if(matrixID>count){return;}
        if(matrixID+start_matrix_ID>N){return;}
        int leftover = i%matrixsize;
        int col = leftover/matrix_dim;
        int row = leftover%matrix_dim;
        if(col!=row){
            dest[start_matrix_ID*matrixsize+i]=0;
        }else{
            dest[start_matrix_ID*matrixsize+i]=defaultvalue[row];
        }
    }
}

__global__ void predict(float*d_state_updated, float* state_predicted, int num_tracks, int N, int n){
    //make prediction based on F and current state at d_state_updated
    //assume start at track 0, for number of current tracks
    //1 thread for 1 tracks
    //X_p =    F  *   X_updated
    //7*N    7*7     7*N
    //all matrix are row major order stored
    //F is simple, so avoid direct matrix multiplication
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>num_tracks){return;}
    for(int i = tid;i<num_tracks;i+=totalthreads){
        if(i>N){return;}
        float x = d_state_updated[i];
        float y = d_state_updated[i+1*N];
        float s = d_state_updated[i+2*N];
        float r = d_state_updated[i+3*N];
        float x_dot = d_state_updated[i+4*N];
        float y_dot = d_state_updated[i+5*N];
        float s_dot = d_state_updated[i+6*N];
        
        state_predicted[i]=x+x_dot;   //cx
        state_predicted[i+1*N]=y+y_dot;   //cy
        state_predicted[i+2*N]=s+s_dot;   //s
        state_predicted[i+3*N]=r;   //r
        state_predicted[i+4*N]=x_dot;   //x.
        state_predicted[i+5*N]=y_dot;   //y.
        state_predicted[i+6*N]=s_dot;   //s.        
    }
}

void set_first_state(tracker &tracker, int num_current_tracks, int num_added_tracks){
    //wrapper function to set the states for first time added detections
    //check tracker's d_Z array and update to states updated
    //5*N row major to n*N row major
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(min((num_added_tracks+255)/256,1), 1, 1 );
    printf("set first state, added tracks: %d\n", num_added_tracks);
	set_first_state_kernel<<<dimGrid,dimBlock>>>(tracker.d_Z,tracker.d_state_updated,
                                tracker.Max_detection,num_current_tracks,num_added_tracks);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new state kernel failed: %s\n", cudaGetErrorString(errora));
    }
	cudaDeviceSynchronize();
}

void set_first_Pcov(tracker &tracker, int current_tracks, int num_added_tracks){
    //wrapper function to set the P covariance matrix states for first time added detections
    //check tracker's d_Z array and update to d_Pcov
    //5*N row major to n*N row major
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(min((num_added_tracks*tracker.n*tracker.n+255)/256,1), 1, 1 );
	set_first_pcov_kernel<<<dimGrid,dimBlock>>>(tracker.d_Pcov,tracker.Max_detection,
                                current_tracks,num_added_tracks,tracker.n);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new Pcov kernel failed: %s\n", cudaGetErrorString(errora));
    }
    cudaDeviceSynchronize();
}

void set_single_F(tracker &tracker){
    //wrapper function to set the F matrix
    //use constant velocity model
    //row major storage
    //1	0	0	0	1	0	0
    //0	1	0	0	0	1	0
    //0	0	1	0	0	0	1
    //0	0	0	1	0	0	0
    //0	0	0	0	1	0	0
    //0	0	0	0	0	1	0
    //0	0	0	0	0	0	1
    float*F = (float*)calloc(tracker.n*tracker.n,sizeof(float));
    for(int i =0;i<tracker.n;i++){F[i*tracker.n+i]=1;}
    F[4]=1;
    F[5+tracker.n]=1;
    F[6+tracker.n*2]=1;
    cudaMemcpy(tracker.d_F,F,sizeof(float)*tracker.n*tracker.n,cudaMemcpyHostToDevice);
    free(F);
}

void set_single_Q(tracker &tracker){
    //wrapper function to set the Q matrix
    //n*n
    //0.01	0	0	0	0	0	0
    //0	   0.01	0	0	0	0	0
    //0	    0  0.01	0	0	0	0
    //0	    0	0 0.001	0	0	0
    //0	    0	0	0	5	0	0
    //0	    0	0	0	0	5	0
    //0	    0	0	0	0	0	5
    float*Q = (float*)calloc(tracker.n*tracker.n,sizeof(float));
    Q[0]=0.01;
    Q[1+tracker.n*1]=0.01;
    Q[2+tracker.n*2]=0.01;
    Q[3+tracker.n*3]=0.001;
    Q[4+tracker.n*4]=5;
    Q[5+tracker.n*5]=5;
    Q[6+tracker.n*6]=5;
    
    cudaMemcpy(tracker.d_Q,Q,sizeof(float)*tracker.n*tracker.n,cudaMemcpyHostToDevice);
    free(Q);
}
void set_single_R(tracker &tracker){
    //wrapper function to set the R matrix
    //m include an extra unit for score
    //but R do not need, only use m-1 as dimension,
    //stored at col-major format for cublas compatibility, but right now no influence
    //m*m
    //0.1	0	0	0
    //0	   0.1	0	0
    //0	    0  0.1	0
    //0	    0	0  0.1
    int m = tracker.m-1;
    float*R = (float*)calloc(m*m,sizeof(float));
    for(int i=0;i<m;i++){
        R[i+m*i]=0.1;
    }
    cudaMemcpy(tracker.d_R,R,sizeof(float)*tracker.m*tracker.m,cudaMemcpyHostToDevice);
    free(R);
}
void set_single_H(tracker &tracker){
    //wrapper function to set the H matrix
    //m*n, stored at col-major format for cublas compatibility
    //only use m-1 as dimension
    //1  0	0	0	0	0	0
    //0	 1	0	0	0	0	0
    //0	 0  1	0	0	0	0
    //0	 0	0   1	0	0	0
    int m = tracker.m-1;
    float*H = (float*)calloc(m*tracker.n,sizeof(float));
    for(int i =0;i<m;i++){H[i*tracker.m+i]=1;}
    
    cudaMemcpy(tracker.d_H,H,sizeof(float)*tracker.m*tracker.n,cudaMemcpyHostToDevice);
    free(H);
}

void make_prediction(tracker &tracker, int num_current_tracks){
    //wrapper function to do predicted states
    //use the states in d_state_updated and d_F matrix to make predictions on d_state_predicted
    //X_p =    F  *   X
    //7*N    7*7     7*N
    //row major order stored
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(min((num_current_tracks+255)/256,1), 1, 1 );
    printf("set predicted state, num of tracks: %d\n", num_current_tracks);
	predict<<<dimGrid,dimBlock>>>(tracker.d_state_updated,tracker.d_state_predicted,
                                num_current_tracks,tracker.Max_Tracks,tracker.n);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("set predicted state kernel failed: %s\n", cudaGetErrorString(errora));
    }
	cudaDeviceSynchronize();

}