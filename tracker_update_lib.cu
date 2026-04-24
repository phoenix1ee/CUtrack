#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include "include/sort_lib.h"

__global__ void set_first_state_kernel(float* source, float* dest, int N, int start_col, int count){
    //source: 5*N row major, each detection sub element in contiguous
    //[cx,cy,w,h]
    //dest: n*N row major, each state factor in contiguous
    //[cx,cy,s,r,x.,y.,s.]
    //1 thread for each states
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>count){return;}
    for(int i = tid;i<count;i+=totalthreads){

        int col = start_col+i;
        if (col>=N)return;
        float w = source[col+2*N];
        float h = source[col+3*N];
        
        dest[col]=source[col];   //cx
        dest[col+N]=source[col+N];   //cy
        dest[col+2*N]=w*h;   //s
        dest[col+3*N]=w/h;   //r
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

void set_single_R(tracker &tracker){

}
void set_single_Q(tracker &tracker){

}
void set_single_H(tracker &tracker){

}
