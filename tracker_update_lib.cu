#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include "include/sort_lib.h"
#include <cublas_v2.h>

__global__ void set_first_state_kernel(float* source, float* dest, int N, int start_col, int count){
    //source: 5*N row major, each detection sub element in contiguous
    //[cx,cy,s,r]
    //dest: n*N row major, each state factor in contiguous
    //[cx,cy,s,r,x.,y.,s.]
    //1 thread for each track
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

__global__ void set_new_track_state_kernel(float* d_Z, float* d_state, int* d_match_detection,
                        int N, int*d_totaltracks, int Num_detection){
    //check the detection match and added unmatched detections as new tracks states
    //1 thread for each track
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>Num_detection){return;}
    for(int i = tid;i<Num_detection;i+=totalthreads){
        int match = d_match_detection[i];
        if(match==-1){
            int col = atomicAdd(d_totaltracks,1);
            if (col>=N)return;
            d_state[col]=d_Z[i];   //cx
            d_state[col+N]=d_Z[i+N];   //cy
            d_state[col+2*N]=d_Z[i+2*N];   //s
            d_state[col+3*N]=d_Z[i+3*N];   //r
            d_state[col+4*N]=1e-3;   //x.
            d_state[col+5*N]=1e-3;   //y.
            d_state[col+6*N]=1e-3;   //s.        
        }
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

__global__ void set_first_age_hit_kernel(int* d_age, int* d_hit_streak, int* d_track_status,
                                        int N, int start_col, int count){
    //set age = 0 and hit streak = 1 for tracks starting start_col for count 
    //set status = 1 for actice but not good
    //1 thread for each track

    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>count){return;}
    for(int i = tid;i<count;i+=totalthreads){
        int col = start_col+i;        
        d_age[col]=0;
        d_hit_streak[col]=1;
        d_track_status[col]=1;
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

__global__ void track_status_update_kernel(int*d_matchtrack,int* d_age, int* d_hit_streak, int* d_track_status,
                                            int* d_active_count, int* d_good_count, int* d_bad_count,
                                           int N, int totaltracks){
    //age> 3 is bad, delete
    //hit > 3 and age <=3 is active, status =2, display
    //when 0<hit<=3, status = 1, active but not display or die
    //1 thread for each track
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int lane = tid&31;
    int good_count = 0;
    int active_count = 0;
    int bad_count = 0;
    //use thread 0 to reset global counter
    if(tid==0){
        atomicExch(d_active_count, 0);
        atomicExch(d_good_count, 0);
        atomicExch(d_bad_count, 0);
    }
    //loop thru each track
    for(int i = tid;i<totaltracks;i+=totalthreads){
        int matchindex = d_matchtrack[i];
        int age = d_age[i];
        int hit = d_hit_streak[i];
        if (matchindex!=-1){hit++;}
        else{age++;}
        int status = -1;
        if (age>3){
            status=0;
            bad_count+=1;
        }else{
            if (hit<=3){
                status=1;
                active_count+=1;
            }else{
                status=2;
                good_count+=1;
            }
        }
        d_track_status[i]=status;
        d_age[i]=age;
        d_hit_streak[i]=hit;
        for (int offset = 16; offset > 0; offset /= 2) {
            good_count += __shfl_down_sync(0xffffffff, good_count, offset);
            active_count += __shfl_down_sync(0xffffffff, active_count, offset);
            bad_count += __shfl_down_sync(0xffffffff, bad_count, offset);
        }
        if(lane==0){
            atomicAdd(d_active_count,active_count);
            atomicAdd(d_good_count,good_count);
            atomicAdd(d_bad_count,bad_count);
        }
    }
}

__global__ void update_track_count_kernel(int* d_active_count, int* d_good_count,
                                        int* d_totaltracks, int* d_goodtracks){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid==0){
        *d_totaltracks=*d_active_count+*d_good_count;
        *d_goodtracks = *d_good_count;
    }
    printf("\ntotal active tracks: %d\n", *d_totaltracks);
}

__global__ void track_output_kernel(float* d_state_update, float*d_state_output, int* d_track_status,int* d_reindex_buffer,
                            int* d_active_count, int* d_good_count, int* d_bad_count,
                            int N,int track_count, float scale, float padx, float pady){
    //process to output buffer and mark down the reindex buffer for future use
    //1 thread for each track
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    //read the count as offset reference to output buffer
    //||good tracks.....  || active tracks......                  ||bad tracks.....||
    //  [0,.good count-1]   [good count,good count+active count]   [good count+active count,good count+active count+bad-1]

    int active_offset = *d_good_count;
    __syncthreads();
    //loop thru each track
    for(int i = tid;i<track_count;i+=totalthreads){
        int status = d_track_status[i];
        int index = -1;
        //grep a slot for status 1 or 2
        if (status==2){
            index = atomicAdd(d_good_count,-1)-1;
        }else if(status==1){
            index = atomicAdd(d_active_count,-1)-1+active_offset;
        }else{
            //status = 0 means delete and skip from reindexing
            index = -1;
        }
        d_reindex_buffer[i]=index;
        //read from states and update the output buffer
        if(index!=-1 && status==2){
            //if not delete state, write to output, convert to original scale
            float cx = d_state_update[i];
            float cy = d_state_update[i+N];
            float s = d_state_update[i+N*2];
            float r = d_state_update[i+N*3];
            float w = sqrt(s*r);
            float h = sqrt(s/r);
            float x1=cx-w/2;
            float x2=cx+w/2;
            float y1=cy-h/2;
            float y2=cy+h/2;
            //original format as d_state
            d_state_output[index] = (x1-padx)/scale;
            d_state_output[index+N] = (y1-pady)/scale;
            d_state_output[index+N*2] = (x2-padx)/scale;
            d_state_output[index+N*3] = (y2-pady)/scale;
        }
    }
}

__global__ void rearrange_track_to_buffer_kernel(int* d_reindex_buffer,
    int* d_track_status, float*d_state_updated,int*d_age, int*d_hit,
    int*d_track_status_buffer,float*d_state_buffer,int*d_age_buffer, int*d_hit_buffer, 
    int N, int n, int totaltracks){
    //kernel to copy data to buffer, tracks status,update states, updated Pcov, age, hit streak
    //before rearrange tracks and related data
    //1 thread for each track
    int totalthreads = gridDim.x*blockDim.x;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>N)return;
    for(int i = tid;i<totaltracks;i+=totalthreads){
        int newindex = d_reindex_buffer[i];
        if(newindex!=-1){
            int status = d_track_status[i];
            int age = d_age[i];
            int hit = d_hit[i];
            d_track_status_buffer[newindex]=status;
            d_age_buffer[newindex]=age;
            d_hit_buffer[newindex]=hit;
            for(int j=0;j<n;j++){
                d_state_buffer[j*N+newindex]=d_state_updated[j*N+i];
            }
        }
    }
}

__global__ void rearrange_Pcov_to_buffer_kernel(int* d_reindex_buffer, float* d_Pcov_update, float* d_Pcov_buffer
                             ,int N, int n, int totaltracks){
    //Copy each Pcov_update to buffer
    //use 1 block for each matrix P
    int blockid = blockIdx.x;
	int gridsize = gridDim.x;
    int blocksize = blockDim.x*blockDim.y;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
	if (blockid >= N) return;
    //use 1 block for each matrix P
    //loop to write to P buffer
    for(int mid = blockid;mid<totaltracks;mid+=gridsize){
        int newindex = d_reindex_buffer[mid];
        if(newindex!=-1){
            for(int j=tid;j<n*n;j+=blocksize){
                float v = d_Pcov_update[mid*n*n+j];
                d_Pcov_buffer[newindex*n*n+j]=v;
            }
        }
    }
}


__global__ void ZminusHX(float* d_Z, float*d_HX, int* d_match_track,
                        int N, int m, int activetracks){
    //kernel to find get matching detections to calculate Z-HX
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
            for(int j = 0;j<m;j++){
                float hx = d_HX[j*N+i];
                d_HX[j*N+i]=d_Z[j*N+matching]-hx;
            }
        }else{
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

__global__ void IminusKH(float* d_Pcov_update, float* d_Pcov_buffer ,int N, int n, int batchCount){
    //calculate I-KH, KH at Pcov_update
    //use 1 block for each matrix P
    int blockid = blockIdx.x;
	int gridsize = gridDim.x;
    int blocksize = blockDim.x*blockDim.y;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
	if (blockid >= N) return;
    //use 1 block for each matrix P
    //loop to write to P
    for(int mid = blockid;mid<batchCount;mid+=gridsize){
        for(int j=tid;j<n*n;j+=blocksize){
            float v = d_Pcov_update[mid*n*n+j];
            float i = 0.0;
            if(j%n==j/n){
                i = 1.0;
            }
            d_Pcov_buffer[mid*n*n+j]=i-v;
        }
    }

}

void set_first_state(tracker &tracker, int num_current_tracks, int num_added_tracks){
    //wrapper function to set the states for first time added detections
    //check tracker's d_Z array and update to states updated
    //5*N row major to n*N row major
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(max((num_added_tracks+255)/256,1), 1, 1 );
	set_first_state_kernel<<<dimGrid,dimBlock>>>(tracker.d_Z,tracker.d_state_updated,
                                tracker.Max_detection,num_current_tracks,num_added_tracks);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new state kernel failed: %s\n", cudaGetErrorString(errora));
    }
}

void set_first_Pcov(tracker &tracker, int current_tracks, int num_added_tracks){
    //wrapper function to set the P covariance matrix states for first time added detections
    //check tracker's d_Z array and update to d_Pcov
    //5*N row major to n*N row major
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(max((num_added_tracks*tracker.n*tracker.n+255)/256,1), 1, 1 );
	set_first_pcov_kernel<<<dimGrid,dimBlock>>>(tracker.d_Pcov,tracker.Max_detection,
                                current_tracks,num_added_tracks,tracker.n);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new Pcov kernel failed: %s\n", cudaGetErrorString(errora));
    }
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

void set_first_age_hit_status(tracker &tracker, int num_current_tracks, int num_added_tracks){
    //wrapper function to set the age = 0 for first time added tracks
    //at d_age
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(max((num_added_tracks+255)/256,1), 1, 1 );
	set_first_age_hit_kernel<<<dimGrid,dimBlock>>>(tracker.d_age,tracker.d_hit_streak,tracker.d_track_status,
                    tracker.Max_detection,num_current_tracks,num_added_tracks);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("setting new age and hit streak kernel failed: %s\n", cudaGetErrorString(errora));
    }
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
	dim3 dimGrid(max((num_current_tracks+255)/256,1), 1, 1 );
	predict_state<<<dimGrid,dimBlock,0>>>(tracker.d_state_updated,tracker.d_state_predicted,
                                num_current_tracks,tracker.Max_Tracks,tracker.n);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("set predicted state kernel failed: %s\n", cudaGetErrorString(errora));
    }
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
    cudaMemcpy(d_P_predict,d_P,sizeof(float)*n*n*totaltracks,cudaMemcpyDeviceToDevice);
    
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
    //use cublas

    float*d_x_predict = tracker.d_state_predicted;
    float*d_x_update = tracker.d_state_updated;
    float*d_H=tracker.d_H;
    float*d_Z=tracker.d_Z;
    float*d_Y=tracker.d_y;
    float* d_K=tracker.d_K;
    int* d_match_track = tracker.d_match_track;
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
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    cudaError_t errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on Hxp cublas: %s\n", cudaGetErrorString(errorb));
    }
    //d_Z-H*x_p
    ZminusHX<<<max((totaltracks+255)/256,1),256>>>(d_Z,d_Y,d_match_track,tracker.Max_Tracks,m,totaltracks);
    cudaDeviceSynchronize();
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
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on K*Y cublas: %s\n", cudaGetErrorString(errorb));
    }
    //d_x+K*Y
    XplusKy<<<max((totaltracks+255)/256,1),256>>>(d_x_update,d_x_predict,tracker.Max_Tracks,n,totaltracks);
    cudaDeviceSynchronize();
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on X_update kernel: %s\n", cudaGetErrorString(errorb));
    }


}

void update_Pcov(tracker &tracker, int num_current_tracks){
    //wrapper function to calculate posterior updated covariance with kalman gain for each matched tracks
    // updated   P       =   (I-        K     *  H   )   *   P
    //           n*n         n*n       n*m      m*n         n*n
    //        symmetric   symmetric   col_maj  col_maj    symmetric
    //use cublas

    float*d_Pcov_predict = tracker.d_Pcov_predict;
    float*d_Pcov_update = tracker.d_Pcov;
    float*d_Pcov_buff = tracker.d_Pcov_buffer;
    float*d_H=tracker.d_H;
    float*d_K=tracker.d_K;
    int n=tracker.n;
    int m=tracker.m-1;
    int totaltracks = num_current_tracks;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    //calculate K*H, n*n
    //result in col major order n*n
    cublasStatus_t cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, m,
                                  &alpha,
                                  d_K, n, n*(m+1),
                                  d_H, m, 0,
                                  &beta,
                                  d_Pcov_update, n, n*n,
                                  totaltracks);
    cudaDeviceSynchronize();
    //writeDevice2DArrayToFile(d_Pcov_update,tracker.Max_Tracks,n,"d_KH.txt");
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    cudaError_t errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on KH cublas: %s\n", cudaGetErrorString(errorb));
    }
    //caculate (I- K*H), KH at Pcov_buff
    IminusKH<<<max((num_current_tracks+255)/256,1),256>>>(d_Pcov_update,d_Pcov_buff,
                                            tracker.Max_Tracks,n,num_current_tracks);
    cudaDeviceSynchronize();
    //writeDevice2DArrayToFile(d_Pcov_buff,tracker.Max_Tracks,n,"d_I-KH.txt");
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on I-KH: %s\n", cudaGetErrorString(errorb));
    }
    //calculate (I-KH)*P, n*n
    cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  d_Pcov_buff, n, n*n,
                                  d_Pcov_predict, n, n*n,
                                  &beta,
                                  d_Pcov_update, n, n*n,
                                  totaltracks);
    cudaDeviceSynchronize();
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on Pcov update cublas: %s\n", cudaGetErrorString(errorb));
    }
}

void update_track_status(tracker &tracker, int num_current_tracks){
    //wrapper to update track age, hit and status of 0,1,2 and count
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(max((num_current_tracks+255)/256,1), 1, 1 );
	track_status_update_kernel<<<dimGrid,dimBlock>>>(tracker.d_match_track,tracker.d_age,tracker.d_hit_streak,tracker.d_track_status,
                    tracker.d_active_count,tracker.d_good_count,tracker.d_bad_count,
                    tracker.Max_detection,num_current_tracks);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("update track status kernel failed: %s\n", cudaGetErrorString(errora));
    }
}

void update_track_count(tracker &tracker){
    int* d_totaltrack = tracker.d_totaltracks;
    int* d_active_count = tracker.d_active_count;
    int* d_good_count = tracker.d_good_count;
    int* d_goodtrack = tracker.d_goodtracks;
    update_track_count_kernel<<<1,1>>>(d_active_count,d_good_count,d_totaltrack,d_goodtrack);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("update device track count kernel failed: %s\n", cudaGetErrorString(errora));
    }
}

void update_output_buffer(tracker &tracker,int num_current_tracks, int w_yolo, int h_yolo, int w_orig, int h_orig){
    //wrapper to update the output buffer with status 2 tracks according track status and states
    //and mark the reindex buffer at d_match_track

    float scale = fminf((float)w_yolo/(float)w_orig,(float)h_yolo/(float)h_orig); 
    float pad_x = ((float)w_yolo-(float)w_orig*scale)/2;
    float pad_y = ((float)h_yolo-(float)h_orig*scale)/2;
    dim3 dimBlock(256, 1, 1 );
	dim3 dimGrid(max((num_current_tracks+255)/256,1), 1, 1 );
	track_output_kernel<<<dimGrid,dimBlock>>>(tracker.d_state_updated,tracker.d_state_output,tracker.d_track_status,
        tracker.d_match_track,tracker.d_active_count,tracker.d_good_count,tracker.d_bad_count,
        tracker.Max_detection,num_current_tracks, scale, pad_x, pad_y);
	cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("output track status kernel failed: %s\n", cudaGetErrorString(errora));
    }
}

void rearrangetracks(tracker &tracker,int num_current_tracks,int num_frame){
    //rearrange tracks status,update states, updated Pcov, age, hit streak
    //and total track count
    //reindex buffer at d_match_track

    float*d_x_buffer = tracker.d_state_predicted;
    float*d_x_update = tracker.d_state_updated;
    float*d_Pcov_buffer = tracker.d_Pcov_buffer;
    float*d_Pcov_update = tracker.d_Pcov;
    int* d_reindex_buffer = tracker.d_match_track;
    int*d_age = tracker.d_age;
    int*d_hit_streak = tracker.d_hit_streak;
    int*d_track_status = tracker.d_track_status;

    int* d_track_status_buffer = (int*)tracker.d_IOU;
    int* d_age_buffer = d_track_status_buffer+tracker.Max_Tracks;
    int* d_hit_streak_buffer = d_age_buffer+tracker.Max_Tracks;
    int n=tracker.n;
    int totaltracks = num_current_tracks;
    //Copy states age hit status to buffer
    rearrange_track_to_buffer_kernel<<<max((totaltracks+255)/256,1),256>>>
    (d_reindex_buffer,d_track_status,d_x_update,d_age,d_hit_streak,
    d_track_status_buffer,d_x_buffer,d_age_buffer,d_hit_streak_buffer,tracker.Max_Tracks,
                        n,totaltracks);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("copy tracks to buffer kernek failed: %s\n", cudaGetErrorString(errora));
    }
    //copy Pcov to buffer
    rearrange_Pcov_to_buffer_kernel<<<totaltracks,256>>>(d_reindex_buffer,d_Pcov_update,d_Pcov_buffer,
        tracker.Max_Tracks,n,totaltracks);
    cudaDeviceSynchronize();
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("copy track cov to buffer kernel failed: %s\n", cudaGetErrorString(errora));
    }
    //Copy buffered states back to states
    //states
    cudaMemcpy(d_x_update,d_x_buffer,sizeof(float)*n*tracker.Max_Tracks,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Pcov_update,d_Pcov_buffer,sizeof(float)*n*n*totaltracks,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_age,d_age_buffer,sizeof(int)*totaltracks,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_hit_streak,d_hit_streak_buffer,sizeof(int)*totaltracks,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_track_status,d_track_status_buffer,sizeof(int)*totaltracks,cudaMemcpyDeviceToDevice);
}

void add_new_tracks(tracker &tracker, int num_detections){
    //add unmatched detection as new tracks
    float*d_Z = tracker.d_Z;
    float*d_state = tracker.d_state_updated;
    int* d_match_detection = tracker.d_match_detections;
    int*d_totaltracks = tracker.d_totaltracks;
    int new_totaltracks = 0;
    int N = tracker.Max_Tracks;

    //add state
    int old_num_tracks;
    cudaMemcpy(&old_num_tracks,d_totaltracks,sizeof(int),cudaMemcpyDeviceToHost);
    set_new_track_state_kernel<<<max(1,(num_detections+255)/256),256>>>
    (d_Z,d_state,d_match_detection,N,d_totaltracks,num_detections);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("add new tracks from unmatched detection kernek failed: %s\n", cudaGetErrorString(errora));
    }
    cudaMemcpy(&new_totaltracks,d_totaltracks,sizeof(int),cudaMemcpyDeviceToHost);
    int added_track = new_totaltracks-old_num_tracks;
    //set Pcov age,hit,status
    set_first_Pcov(tracker,old_num_tracks,added_track);
    set_first_age_hit_status(tracker,old_num_tracks,added_track);
}