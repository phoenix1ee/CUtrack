#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include "include/sort_lib.h"
#include <cublas_v2.h>

__device__ void CopyTrack(float*d_state_update, float* d_pcov, int* d_age, int* hit_streak,
                         int n, int trackIDsource, int trackIDdest){
    for(int i=0;i<n*n;i++){

    }
}

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
        /*
        defaultvalue[0]=1.0;
        defaultvalue[1]=1.0;
        defaultvalue[2]=1.0;
        defaultvalue[3]=0.01;
        defaultvalue[4]=10.0;
        defaultvalue[5]=10.0;
        defaultvalue[6]=1.0;
        */
        defaultvalue[0]=10.0;
        defaultvalue[1]=10.0;
        defaultvalue[2]=10.0;
        defaultvalue[3]=0.1;
        defaultvalue[4]=2.0;
        defaultvalue[5]=2.0;
        defaultvalue[6]=4.0;
        
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

__global__ void set_first_age_hit_kernel(int* d_age, int* d_hit_streak, int N, int start_col, int count){
    //set age = 0 and hit streak = 0 for tracks starting start_col for count 
    //1 thread for each track

    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>count){return;}
    for(int i = tid;i<count;i+=totalthreads){
        int col = start_col+i;        
        d_age[col]=0;
        d_hit_streak[col]=0;
    }
}

__global__ void predict_state(float*d_state_updated, float* state_predicted, int num_tracks, int N, int n){
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

__global__ void hit_age_delete_kernel(float* d_state_predict, float* d_state_update){

}

__global__ void ZminusHX(float* d_Z, float*d_HX, int* d_match_track,int* d_age, int* d_hit_streak,
                        int N, int m, int activetracks){
    //HX is in d_Y
    //calculate Z-HX
    //both row major order m*N
    int totalsize  = blockDim.x*gridDim.x;
    int global_id = blockIdx.x*blockDim.x+threadIdx.x;
    if(global_id>N)return;
    //for each track i
    for(int i = global_id;i<activetracks;i+=totalsize){
        //for each measurement element j in Z at track i
        int matching = d_match_track[i];
        if(matching!=-1){
            d_hit_streak[i]+=1;
            for(int j = 0;j<m;j++){
                float hx = d_HX[j*N+i];
                d_HX[j*N+i]=d_Z[j*N+matching]-hx;
            }
        }else{
            d_age[i]+=1;
            for(int j = 0;j<m;j++){
                d_HX[j*N+i]=0.0;
            }
        }
    }
}

__global__ void XplusKy(float* d_state_update,float*d_state_predict, 
                        int N, int n, int activetracks){
    //KY is in d_state_update
    //calculate X_predict+KY
    //both row major order m*N
    int totalsize  = blockDim.x*gridDim.x;
    int global_id = blockIdx.x*blockDim.x+threadIdx.x;
    if(global_id>N)return;
    //for each track i
    for(int i = global_id;i<activetracks;i+=totalsize){
        for(int j = 0;j<n;j++){
            float ky = d_state_update[j*N+i];
            d_state_update[j*N+i]=d_state_predict[j*N+i]+ky;
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

void set_first_age_hit(tracker &tracker, int num_current_tracks, int num_added_tracks){
    //wrapper function to set the age = 0 for first time added tracks
    //at d_age
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(min((num_added_tracks+255)/256,1), 1, 1 );
    printf("set first age=0 and hit streak=0 for added tracks: %d\n", num_added_tracks);
	set_first_age_hit_kernel<<<dimGrid,dimBlock>>>(tracker.d_age,tracker.d_hit_streak,tracker.Max_detection,
                                                num_current_tracks,num_added_tracks);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new age and hit streak kernel failed: %s\n", cudaGetErrorString(errora));
    }
	cudaDeviceSynchronize();
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
    for(int i =0;i<m;i++){H[i*m+i]=1;}
    
    cudaMemcpy(tracker.d_H,H,sizeof(float)*tracker.m*tracker.n,cudaMemcpyHostToDevice);
    free(H);
}

void make_state_prediction(tracker &tracker, int num_current_tracks){
    //wrapper function to do predicted states and covariance
    //use the states in d_state_updated and d_F matrix to make predictions on d_state_predicted
    //X_p =    F  *   X
    //7*N    7*7     7*N
    //row major order stored

    //cudaStream_t stream_predict_state;
    //cudaStreamCreate(&stream_predict_state);

    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(min((num_current_tracks+255)/256,1), 1, 1 );
    printf("set predicted state, num of tracks: %d\n", num_current_tracks);
	predict_state<<<dimGrid,dimBlock,0>>>(tracker.d_state_updated,tracker.d_state_predicted,
                                num_current_tracks,tracker.Max_Tracks,tracker.n);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("set predicted state kernel failed: %s\n", cudaGetErrorString(errora));
    }
    cudaDeviceSynchronize();
}

void make_cov_prediction(tracker &tracker, int num_current_tracks){
    //wrapper function to do predicted states and covariance
    //use the cov matrix in d_Pcov and d_F matrix to make predictions on d_Pcov_predict
    //current implementation use n=7
    // N P_cov matrix
    //P_predict =   F  *  P  *  F_T  +   Q
    //7*7          7*7   7*7    7*7     7*7
    //P,Q is symmetrical but F is row major stored
    
    //use cublas

    float*d_P = tracker.d_Pcov; //strided between d_Pcov is n=7*n=7
    float*d_P_predict = tracker.d_Pcov_predict; //strided between d_Pcov_predict is n=7*n=7
    float*d_F=tracker.d_F;
    float*d_Q=tracker.d_Q;
    int n=tracker.n;
    int totaltracks = num_current_tracks;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    //cublasSetStream(handle, stream);

    //calculate PF^T, n*n  F is row major, so no need CUBLAS_OP_T for F
    cublasStatus_t cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  d_P, n, n*n,
                                  d_F, n, 0,
                                  &beta,
                                  d_P_predict, n, n*n,
                                  totaltracks);
    cudaDeviceSynchronize();
    //writeDevice2DArrayToFile(d_P_predict,n,n,"d_PF_T.txt");
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    cudaError_t errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on PFt cublas: %s\n", cudaGetErrorString(errorb));
    }
    
    //calculate F*PF^T, n*n , F is row major, so need CUBLAS_OP_T for F
    cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  d_F, n, 0,
                                  d_P_predict, n, n*n,
                                  &beta,
                                  d_P, n, n*n,
                                  totaltracks);
    cudaDeviceSynchronize();
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    //calculate FPF^T+Q, n*n
    MMAdd1toMany<<<totaltracks,64,n*n*sizeof(float)>>>(d_P,d_Q,n*n,n,n,totaltracks);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("CUDA error on F*PFt cublas: %s\n", cudaGetErrorString(errora));
    }
    writeDevice2DArrayToFile(d_P,n,n,"d_P_predict.txt");
    cudaMemcpy(d_P_predict,d_P,sizeof(float)*n*n*totaltracks,cudaMemcpyDeviceToDevice);
    //cudaMemcpyAsync(d_P_predict,d_P,sizeof(float)*n*n*totaltracks,cudaMemcpyDeviceToDevice,stream);
    
    cublasDestroy(handle);
    //cudaStreamDestroy(stream);
}

void update_states_Kalman(tracker &tracker, int num_current_tracks){
    //wrapper function to calculate posterior updated states with kalman gain for each matched tracks
    // innovation y = d_Z   -   H      *    x_predicted
    //                m*M      m*n             n*N
    //               row_mj   col_mj         row_maj
    // updated   x = x_predicted  +   K     *     y
    //           n*N      n*N        n*m         m*N
    //          row_mj   row_mj    col_maj       row major n*M or n*N
    //also update hit streaks and age of matched and unmatch tracks
    //use cublas

    float*d_x_predict = tracker.d_state_predicted;
    float*d_x_update = tracker.d_state_updated;
    float*d_H=tracker.d_H;
    float*d_Z=tracker.d_Z;
    float*d_Y=tracker.d_y;
    float* d_K=tracker.d_K;
    int* d_match_track = tracker.d_match_track;
    int*d_age = tracker.d_age;
    int*d_hit_streak = tracker.d_hit_streak;
    int n=tracker.n;
    int m=tracker.m-1;
    int totaltracks = num_current_tracks;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //calculate H*x_p, m*N
    //need result in row major order m*N
    //calculate x_pT*HT instead

    cublasStatus_t cuA = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  tracker.Max_Tracks, m, n,
                                  &alpha,
                                  d_x_predict, tracker.Max_Tracks,
                                  d_H, m,
                                  &beta,
                                  d_Y, tracker.Max_Tracks);

    cudaDeviceSynchronize();
    writeDevice2DArrayToFile(d_Y,m,tracker.Max_Tracks,"d_HXP.txt");
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    cudaError_t errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on Hxp cublas: %s\n", cudaGetErrorString(errorb));
    }
    //d_Z-H*x_p
    //and hit streak and age update
    ZminusHX<<<(totaltracks+255)/256,256>>>(d_Z,d_Y,d_match_track,d_age,d_hit_streak,tracker.Max_Tracks,m,totaltracks);
    cudaDeviceSynchronize();
    writeDevice2DArrayToFile(d_Y,m,tracker.Max_Tracks,"d_Y.txt");
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on innovation Z-Hxp kernel: %s\n", cudaGetErrorString(errorb));
    }
    //calculate K*Y
    //need result in row major order n*N
    //calculate YT*KT instead
    
    cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  1, n, m,
                                  &alpha,
                                  d_Y,tracker.Max_Tracks, 1,
                                  d_K, n, n*(m+1),
                                  &beta,
                                  d_x_update, tracker.Max_Tracks, 1,
                                  totaltracks);
    cudaDeviceSynchronize();
    writeDevice2DArrayToFile(d_x_update,n,tracker.Max_Tracks,"d_KY.txt");
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on K*Y cublas: %s\n", cudaGetErrorString(errorb));
    }
    //d_x+K*Y
    XplusKy<<<(totaltracks+255)/256,256>>>(d_x_update,d_x_predict,tracker.Max_Tracks,n,totaltracks);
    cudaDeviceSynchronize();
    writeDevice2DArrayToFile(d_x_update,n,tracker.Max_Tracks,"d_X_update.txt");
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on X_update kernel: %s\n", cudaGetErrorString(errorb));
    }


}

void cleantracks(tracker &tracker, int num_current_tracks){
    //increase hit streaks for matched tracks

    //increase age for unmatched tracks

    //check delete threshold and delete

}