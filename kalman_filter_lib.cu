#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "include/helper.h"
#include "include/sort_lib.h"


__global__ void MMAdd1toMany(float* batchedA, float* singleB, int strideA, 
                            int row, int col, int batchCount){
    int blockid = (gridDim.x*blockIdx.y) + blockIdx.x;
	int gridsize = gridDim.x*gridDim.y;
	if (blockid >= batchCount) return;
    //use 1 block for each matrix A
    //use shared mem to store the single matrix, size row*col declared at kernel launch
    extern __shared__ float B[];
    int blocksize = blockDim.x*blockDim.y;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    for(int i=tid;i<row*col;i+=blocksize){
        B[i]=singleB[i];
    }
    //loop to write to A
    for(int mid = blockid;mid<batchCount;mid+=gridsize){
        for(int j=tid;j<row*col;j+=blocksize){
            //printf("HPHT is %.2f\n",batchedA[mid*row*col+j]);
            batchedA[mid*strideA+j]+=B[j];
        }
    }
}

__global__ void print_device_matrix_kernel(float*d_input,int cols,int rows){
    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
				printf("%6.3f ",d_input[j*rows+i]);
		}
		printf("\n");
	}
}
void print_device_matrix_colmajor_from_host(float*d_input,int cols,int rows){
    print_device_matrix_kernel<<<1,1>>>(d_input,cols,rows);
    cudaDeviceSynchronize();
}


void tracker_kalman_gain(tracker* trackerA, int totaltracks, cublasHandle_t handle, cusolverDnHandle_t k_handle){
    //for use with custom Struct Tracker
    float*d_S = trackerA->d_S; //strided between d_S is m=5*m=5

    float*d_PHT=trackerA->d_K; //strided between d_K is m=5*n
    float*d_P=trackerA->d_Pcov_predict;
    float*d_H=trackerA->d_H;
    float*d_R=trackerA->d_R;
    int m = trackerA->m-1; //m is allocated with 1 more unit in tracker object.
    //but only 4 are used as m. So matrix dimension m =4, but strided between d_K is m=5*n
    //d_H and d_R is fine because they are single matrix and declared as m=4, with unused bytes at the end
    int n=trackerA->n;
    
    //solve for Kalman gain for a batch of P, H and R
    //P=n*n H=m*n R=m*m   for totaltracks no. of tracks
    //K = P*H^T*S^-1,  S=H*P*H^T+R
    //All matrix
    //P_k|k-1 = Predicted estimate covariance
    //H_k, R_k

    //PHT version

    const float alpha = 1.0f;
    const float beta = 0.0f;

    //cublasHandle_t handle;
    //cublasCreate(&handle);

    //calculate PH^T, n*m
    cublasStatus_t cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n, m, n,
                                  &alpha,
                                  d_P, n, n*n,
                                  d_H, m, 0,
                                  &beta,
                                  d_PHT, n, (m+1)*n,
                                  totaltracks);
    cudaDeviceSynchronize();
    //writeDevice2DArrayToFile(d_PHT,m,n,"d_PHT.txt");
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    cudaError_t errorb = cudaGetLastError();
    if (errorb != cudaSuccess){
        printf("CUDA error on PHt cublas: %s\n", cudaGetErrorString(errorb));
    }
    
    //calculate H*PH^T, m*m
    cuA = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, m, n,
                                  &alpha,
                                  d_H, m, 0,
                                  d_PHT, n, n*(m+1),
                                  &beta,
                                  d_S, m, (m+1)*(m+1),
                                  totaltracks);
    cudaDeviceSynchronize();
    if (cuA!=CUBLAS_STATUS_SUCCESS){
        printf("sgemm error");
    }
    //calculate HPH^T+R, m*m
    MMAdd1toMany<<<totaltracks,64,m*m*sizeof(float)>>>(d_S,d_R,(m+1)*(m+1),m,m,totaltracks);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(errora));
    }
    //writeDevice2DArrayToFile(d_S,m,m,"d_S.txt");

    //cusolverDnHandle_t k_handle = NULL;
    //cusolverStatus_t cusolver_status;

    //cusolverDnCreate(&k_handle);

    //find out pointers to each matrix S and PHT for batch operations
    float** d_each_d_PHT = trackerA->d_each_K;
    float** d_each_d_S = trackerA->d_each_S;

    //float* temp = (float*)malloc(sizeof(float)*1);
    //cudaMemcpy(temp,d_S+1,sizeof(float)*1,cudaMemcpyDeviceToHost);
    //printf("S found is %.3f\n",*temp);
    //free(temp);
    // info array for factorization operation
    int* info = (int*)malloc(sizeof(int)*totaltracks);
    int* d_info = trackerA->d_info;  

    // Perform Cholesky Factorization in batch for each S to find L: S = L * L^T
    // S = m*m
    cusolverStatus_t cusolver_status = cusolverDnSpotrfBatched(k_handle,
                            CUBLAS_FILL_MODE_LOWER,
                            m,
                            d_each_d_S,
                            m,
                            d_info, 
                            totaltracks);
    cudaDeviceSynchronize();
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("K factorization cublas failed: %s\n", cudaGetErrorString(errora));
    }
    if (cusolver_status!=CUSOLVER_STATUS_SUCCESS){
        printf("factorization error");
    }
    cudaMemcpy(info,d_info,sizeof(int)*totaltracks,cudaMemcpyDeviceToHost);
    //check status of factorization
    for(int i =0;i<totaltracks;i++){
        if (info[i] != 0){
            printf("factorization solver failed, at matrix %d with code %d\n", i,info[i]);
            return;
        }
    }
    free(info);
    //cusolverDnDestroy(k_handle);
    
    // Solve for K:  K * (L*L^T) = (PH^T)   /    KS = (PH^T)
    // S is symmetric, tell solver to use lower half
    // d_S contains L, m*m  PH^T = n*m

    // run the first cublasStrsmBatched to solve for KL in batch, x * L^T = (PH^T)
    cublasStrsmBatched(handle,
                       CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        n,m,
                        &alpha,
                        d_each_d_S,m,
                        d_each_d_PHT,n,
                        totaltracks);    
    cudaDeviceSynchronize();

    // run the second cublasStrsmBatched to solve for K in batch, K * L = x
    cublasStrsmBatched(handle,
                       CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                        n,m,
                        &alpha,
                        d_each_d_S,m,
                        d_each_d_PHT,n,
                        totaltracks);    
    cudaDeviceSynchronize();
    //d_PHT now contains kalman gain matrix n*m.
    //writeDevice2DArrayToFile(d_PHT,m,n,"d_K.txt");

    //cublasDestroy(handle);
    }
