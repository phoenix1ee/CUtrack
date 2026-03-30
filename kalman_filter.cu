#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "helper.h"
#include "sort_lib.h"


void cublasTranspose_simple(float* d_in, float* d_out, int o_row, int o_col){
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t T_handle;
    cublasCreate(&T_handle);

    cublasStatus_t T_status=CUBLAS_STATUS_SUCCESS;

    //transpose the matrix from o_row*o_col to o_col to o_row
    cublasSgeam(T_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                o_col,o_row,
                &alpha,
                d_in, o_row,
                &beta,
                nullptr, o_col,
                d_out, o_col);

    cudaDeviceSynchronize();
    if (T_status!=CUBLAS_STATUS_SUCCESS){
        printf("cublas geam to transpose failed");
    }
    cublasDestroy(T_handle);
}

void cublasadd_simple(float* d_A, float* d_B, int n){
    //wrapper function to do simple add with cublasSaxpy
    const float alpha = 1.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;

    // A is n*n matrix
    // B is nxn matrix
    // compute: B=A+B
    status = cublasSaxpy(handle, n*n,
                &alpha, 
                d_A, 1,           
                d_B, 1);          

    cudaDeviceSynchronize();

    if (status!=CUBLAS_STATUS_SUCCESS){
        printf("cublas add failed");
    }

    cublasDestroy(handle);
}

void cublasmmulti(float* d_A, float* d_B, float* d_C, bool A_T, bool B_T, int m, int n, int k) {
    //wrapper function to do simple matrix multiplication with cublas
    //do simple C=A*B
    //A=m*k
    //B=k*n
    //check transpose needs
    cublasOperation_t A_op = CUBLAS_OP_N;
    if (A_T){
        A_op = CUBLAS_OP_T;
    }
    cublasOperation_t B_op = CUBLAS_OP_N;
    if (B_T){
        B_op = CUBLAS_OP_T;
    }


    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;

    // A is mxk matrix
    // B is kxn matrix
    // compute: C=A*B
    status = cublasSgemm_v2(handle, 
                A_op, B_op, 
                m, n, k, 
                &alpha, 
                d_A, m,           
                d_B, n,           
                &beta, 
                d_C, m);          

    cudaDeviceSynchronize();

    if (status!=CUBLAS_STATUS_SUCCESS){
        printf("cublas gemm failed");
    }

    cublasDestroy(handle);
}

void kalman_gain_single(float* d_K, float*d_P,float*d_H,float*d_R,int m, int n){
    //solve for Kalman gain from P, H and R
    //P=n*n H=m*n R=m*m
    //K = P*H^T*S^-1,  S=H*P*H^T+R
    //All matrix
    //P_k|k-1 = Predicted estimate covariance
    //H_k, R_k
    
    float* d_HP;
    cudaMalloc((void**)&d_HP, sizeof(float)*n*m);
    float* d_S;
    cudaMalloc((void**)&d_S, sizeof(float)*m*m);

    //calculate H*P, m*n
    cublasmmulti(d_H,d_P,d_HP,false,false,m,n,n);

    //calculate (HP)*H^T, m*m
    cublasmmulti(d_HP,d_H,d_S,false,true,m,m,n);

    //calculate HPH^T+R, m*mS
    cublasadd_simple(d_R,d_S,m*m);

    
    cusolverDnHandle_t k_handle = NULL;
    cusolverDnParams_t params = NULL;
    //cusolverStatus_t cusolver_status;

    cusolverDnCreate(&k_handle);
    cusolverDnCreateParams(&params);

    size_t h_workspace=0;
    size_t d_workspace=0;
    
    //create workspace
    cusolverStatus_t cusolver_status = cusolverDnXpotrf_bufferSize(
        k_handle, params, CUBLAS_FILL_MODE_LOWER,
        m, CUDA_R_32F,
        d_S, 
        m, CUDA_R_32F,
        &d_workspace, &h_workspace);
    cudaDeviceSynchronize();
    
    if (cusolver_status!=CUSOLVER_STATUS_SUCCESS){
        printf("workspace allocation error");
    }
    
    // Perform Cholesky Factorization to find L: S = L * L^T
    // S = m*m
    void* h_work = (void*)malloc(h_workspace * sizeof(float));
    void* d_work;
    cudaMalloc((void**)&d_work, sizeof(float)*d_workspace);
    int info = 0;
    int* d_info;   
    cudaMalloc((void**)&d_info, sizeof(int));
    
    cusolverDnXpotrf(
        k_handle, params, CUBLAS_FILL_MODE_LOWER,
        m, CUDA_R_32F,
        d_S,
        m, CUDA_R_32F,
        d_work, d_workspace,
        h_work, h_workspace,
        d_info);
    cudaDeviceSynchronize();
    cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);

    if (info != 0){
        printf("factorization solver failed, %d \n", info);
        return;
    }
    
    // Solve for K:  K * (L*L^T) = (PH^T)   /    KS = (PH^T)
    // S^T*K^T = H*P^T = H*P because P is symmetric
    // S is also symmetric, tell solver to use lower half
    // d_S contains L, m*m  H*P^T = m*n
    // result from solver is K^T, m*n
    cusolverDnXpotrs(
        k_handle, params, CUBLAS_FILL_MODE_LOWER,
        m, n, 
        CUDA_R_32F, d_S, m,
        CUDA_R_32F, d_HP, m,
        d_info);
    cudaDeviceSynchronize();
    cudaMemcpy(&info,d_info,sizeof(int),cudaMemcpyDeviceToHost);
    if (info != 0){
        printf("linear equations solver failed, error code %d \n", info);
        return;
    }
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(k_handle);

    //transpose the K^T to K for output , from m*n to n*m

    cublasTranspose_simple(d_HP,d_K,m,n);

    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_S);
    cudaFree(d_HP);
    free(h_work);
    
}

void predict_positions(float* d_F, float* d_x, int num_objects) {
    //wrapper function to do position prediction for kalman filter using cublas primitives
    //simplied with no control input

    // F is 4x4 state transition matrix
    // x is 4xn object states matrix for n objects
    // compute: x_new = F * x_old
    cublasmmulti(d_F,d_x,d_x,false,false,4,num_objects,4);
}

void kalman_gain_batch(bool*inactive, float*d_PHT, float*d_P,float*d_H,float*d_R,int totaltracks, int m, int n){
    //solve for Kalman gain for a batch of P, H and R
    //P=n*n H=m*n R=m*m   for totaltracks no. of tracks
    //K = P*H^T*S^-1,  S=H*P*H^T+R
    //All matrix
    //P_k|k-1 = Predicted estimate covariance
    //H_k, R_k

    //PHT version
    float* d_S;
    cudaMalloc((void**)&d_S, sizeof(float)*m*m*totaltracks);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    //calculate PH^T, n*m
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n, m, n,
                                  &alpha,
                                  d_P, n, n*n,
                                  d_H, m, m*n,
                                  &beta,
                                  d_PHT, n, m*n,
                                  totaltracks);

    //calculate H*PH^T, m*m
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, m, n,
                                  &alpha,
                                  d_H, m, m*n,
                                  d_PHT, n, n*m,
                                  &beta,
                                  d_S, m, m*m,
                                  totaltracks);

    //calculate HPH^T+R, m*m
    cublasSaxpy(handle, m*m*totaltracks,
                &alpha, 
                d_R, 1,           
                d_S, 1);    

    cusolverDnHandle_t k_handle = NULL;
    //cusolverStatus_t cusolver_status;

    cusolverDnCreate(&k_handle);

    //find out pointers to each matrix S and PHT for batch operations
    float** each_d_PHT = (float**)malloc(sizeof(float*)*totaltracks);
    float** each_d_S = (float**)malloc(sizeof(float*)*totaltracks);
    for (int i =0;i<totaltracks;i++){
        each_d_PHT[i] = d_PHT+i*n*m;
        each_d_S[i]   = d_S+i*m*m;
    }

    //allocate and copy to device for the pointers to sub matrix
    float** d_each_d_PHT;
    cudaError_t errorP = cudaMalloc((void**)&d_each_d_PHT, sizeof(float*) * totaltracks);
    if (errorP != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errorP));
    }
    errorP = cudaMemcpy(d_each_d_PHT,each_d_PHT,sizeof(float*)*totaltracks,cudaMemcpyHostToDevice);
    if (errorP != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errorP));
    }

    float** d_each_d_S;
    cudaError_t errorS = cudaMalloc((void**)&d_each_d_S, sizeof(float*) * totaltracks);
    if (errorS != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errorS));
    }
    errorS = cudaMemcpy(d_each_d_S,each_d_S,sizeof(float*)*totaltracks,cudaMemcpyHostToDevice);
    if (errorS != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errorS));
    }

    // info array for factorization operation
    int* info = (int*)malloc(sizeof(int)*totaltracks);
    int* d_info;   
    cudaMalloc((void**)&d_info, sizeof(int)*totaltracks);

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
    cudaFree(d_info);
    free(info);
    cusolverDnDestroy(k_handle);
    
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
    //d_PHT no contains kalman gain matrix n*m.

    cublasDestroy(handle);

    cudaFree(d_S);

    }
