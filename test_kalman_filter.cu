#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "helper.h"
#include "sort_lib.h"

void test_kalman_gain_single(void){
    //test for kalman gain calculation function
    //final K = {0.95238,0.23810}
    float* K=(float*)malloc(sizeof(float)*2);
    float P[4]={2,0.5,0.5,1};
    float H[4]={1,0};
    float R[4]={0.1};

    
    printf("P:\n");
    printmatrix_colmajor(P,2,2);
    printf("H:\n");
    printmatrix_colmajor(H,2,1);
    printf("R:\n");
    printmatrix_colmajor(R,1,1);

    float*d_K;
    float*d_P;
    float*d_H;
    float*d_R;
    
    cudaMalloc((void**)&d_K,sizeof(float)*2);
    cudaMalloc((void**)&d_P,sizeof(float)*4);
    cudaMalloc((void**)&d_H,sizeof(float)*2);
    cudaMalloc((void**)&d_R,sizeof(float)*1);

    cudaMemcpy(d_P,P,sizeof(float)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_H,H,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_R,R,sizeof(float)*1,cudaMemcpyHostToDevice);

    //kalman calculation
    kalman_gain_single(d_K,d_P,d_H,d_R,1,2);

    cudaMemcpy(K,d_K,sizeof(float)*2,cudaMemcpyDeviceToHost);

    cudaFree(d_K);
    cudaFree(d_P);
    cudaFree(d_H);
    cudaFree(d_R);

    printf("output kalman gain:\n");
    printmatrix_colmajor(K,1,2);

    free(K);
}

void test_kalman_gain_batch(void){
    //test for kalman gain calculation in batch
    //final K = {0.95238,0.23810}
    
    int N = 4;
    printf("total number of tracks: %d \n",N);
    int n = 2;
    int m = 1;

    bool* inactive=(bool*)calloc(N,sizeof(bool));

    float* K=(float*)malloc(sizeof(float)*n*m*N);
    float* P=(float*)malloc(sizeof(float)*n*n*N);
    float* H=(float*)malloc(sizeof(float)*m*n*N);
    float* R=(float*)malloc(sizeof(float)*m*m*N);
    
    for(int i=0;i<N;i++){
        P[i*n*n]=2.0;
        P[i*n*n+1]=0.5;
        P[i*n*n+2]=0.5;
        P[i*n*n+3]=1.0;
    }
    for(int i=0;i<N;i++){
        H[i*m*n]=1.0;
        H[i*m*n+1]=0;
    }
    for(int i=0;i<N;i++){
        R[i*m*m]=0.1;
    }

    printf("state covariance matrix P(use same value for each track for testing): \n");
    printmatrix_colmajor(P,n,n);
    printf("state-to-measurement matrix H: \n");
    printmatrix_colmajor(H,n,m);
    printf("measurement covariance matrix R: \n");
    printmatrix_colmajor(R,m,m);

    float*d_K_all;
    float*d_P_all;
    float*d_H_all;
    float*d_R_all;
    float*d_S_all;
    cudaMalloc((void**)&d_S_all,sizeof(float)*m*m*N);

    cudaMalloc((void**)&d_K_all,sizeof(float)*n*m*N);
    cudaMalloc((void**)&d_P_all,sizeof(float)*n*n*N);
    cudaMalloc((void**)&d_H_all,sizeof(float)*m*n*N);
    cudaMalloc((void**)&d_R_all,sizeof(float)*m*m*N);

    cudaMemcpy(d_P_all,P,sizeof(float)*n*n*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_H_all,H,sizeof(float)*m*n*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_all,R,sizeof(float)*m*m*N,cudaMemcpyHostToDevice);

    //kalman calculation
    kalman_gain_batch(inactive,d_S_all,d_K_all,d_P_all,d_H_all,d_R_all,N,m,n);

    cudaMemcpy(K,d_K_all,sizeof(float)*n*m*N,cudaMemcpyDeviceToHost);
    cudaFree(d_K_all);
    cudaFree(d_P_all);
    cudaFree(d_H_all);
    cudaFree(d_R_all);
    cudaFree(d_S_all);

    printf("output kalman gain K=P*H^T*S^(-1),  S=H*P*H^T+R:\n");
    for (int i=0;i<N;i++){
        printf("track %d :\n",i);
        printmatrix_colmajor(K+i*m*n,m,n);
    }
    free(K);
    free(P);
    free(H);
    free(R);
    
}

void test_predict_positions(void){
    //test for position prediction function
    int num_obj = 4;

    float* x=create_2d_array(num_obj,4);
    printf("input speed vector:\n");
    printmatrix_colmajor(x,num_obj,4);
    float T[4*4]={0.2,0.3,0.4,0.5,0.2,0.3,0.4,0.5,0.2,0.3,0.4,0.5,0.2,0.3,0.4,0.5};
    printf("input transition matrix:\n");
    printmatrix_colmajor(T,4,4);
    cublasHandle_t handle;
    cublasCreate(&handle);

    float*x_d;
    float*T_d;
    cudaMalloc((void**)&x_d,sizeof(float)*num_obj*4);
    cudaMalloc((void**)&T_d,sizeof(float)*16);
    cudaMemcpy(x_d,x,sizeof(float)*num_obj*4,cudaMemcpyHostToDevice);
    cudaMemcpy(T_d,T,sizeof(float)*16,cudaMemcpyHostToDevice);

    //predict_positions(handle,T_d,x_d,num_obj);
    predict_positions(T_d,x_d,num_obj);

    cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf(" kernel failed");
	}
    cudaMemcpy(x,x_d,sizeof(float)*num_obj*4,cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(T_d);
    printf("output transition matrix:\n");
    printmatrix_colmajor(x,num_obj,4);
    cublasDestroy(handle);
    free(x);
}

void test_add_single(void){
    //test matrix add

    float A[4]={1,1,1,1};
    float B[4]={2,2,2,2};

    
    printf("A:\n");
    printmatrix_colmajor(A,2,2);
    printf("B:\n");
    printmatrix_colmajor(B,2,2);

    float*d_A;
    float*d_B;
    
    cudaMalloc((void**)&d_A,sizeof(float)*4);
    cudaMalloc((void**)&d_B,sizeof(float)*4);

    cudaMemcpy(d_A,A,sizeof(float)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*4,cudaMemcpyHostToDevice);

    //kalman calculation
    cublasadd_simple(d_A,d_B,2*2);

    cudaMemcpy(B,d_B,sizeof(float)*4,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    printf("output add:\n");
    printmatrix_colmajor(B,2,2);


}

void test_transpose(void){
    //test matrix transpose

    float A[6]={1,2,1,2,1,2};
    printf("A:\n");
    printmatrix_colmajor(A,3,2);

    float*d_A;
    float*d_B;
    
    cudaMalloc((void**)&d_A,sizeof(float)*6);
    cudaMalloc((void**)&d_B,sizeof(float)*6);

    cudaMemcpy(d_A,A,sizeof(float)*6,cudaMemcpyHostToDevice);

    //kalman calculation
    cublasTranspose_simple(d_A,d_B,2,3);

    cudaMemcpy(A,d_B,sizeof(float)*6,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    printf("output transpose:\n");
    printmatrix_colmajor(A,2,3);
}

void test_MMaddKernel(void){
    float* A;
    A = (float*)malloc(sizeof(float)*18);
    for(int i=0;i<18;i++){
        A[i]=i/6;
    }
    printmatrix_colmajor(A,3,6); 
    float B[6]={1,2,3,4,5,6};
    float* d_A;
    float* d_B;
    cudaMalloc((void**)&d_A,sizeof(int)*18);
    cudaMalloc((void**)&d_B,sizeof(int)*6);
    cudaMemcpy(d_A,A,sizeof(float)*18,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*6,cudaMemcpyHostToDevice);
    MMAdd1toMany<<<4,4,6>>>(d_A,d_B,3,2,3);
    cudaMemcpy(A,d_A,sizeof(float)*18,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    printmatrix_colmajor(A,3,6);    
    free(A);
}

void test_tracker_kalman_gain(void){
    //final K = {0.95238,0.23810}
    //set a max expected detection 2000
    
    int N=2000;
    int Max_detection = 2000;
    //set state variable and detection spec
    int m=1;     //detection
    int n=2;     //state variable

    printf("No of tracks tested=%d\n",N);

    tracker tracker1(N,Max_detection,m,n);
    tracker1.allocateOnDevice();
    tracker* d_tracker;
    cudaMalloc(&d_tracker,sizeof(tracker));
    cudaMemcpy(d_tracker,&tracker1,sizeof(tracker),cudaMemcpyHostToDevice);
    
    KSaddrInitialize<<<(N+255)/256,256>>>(d_tracker);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(errora));
    }

    //H P and R initialize for test
    float H[2]={1.0,0.0};
    float R[1]={0.1};
    float* P=(float*)malloc(sizeof(float)*n*n*N);
    for(int i=0;i<N;i++){
        P[i*n*n]=2.0;
        P[i*n*n+1]=0.5;
        P[i*n*n+2]=0.5;
        P[i*n*n+3]=1.0;
    }
    printf("H matrix:\n");
    printmatrix_colmajor(H,2,1);
    printf("R matrix:\n");
    printmatrix_colmajor(R,1,1);
    printf("P matrix:\n");
    printmatrix_colmajor(P,2,2);


    cudaMemcpy(tracker1.d_H,H,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(tracker1.d_R,R,sizeof(float)*1,cudaMemcpyHostToDevice);
    cudaMemcpy(tracker1.d_Pcov,P,sizeof(float)*n*n*N,cudaMemcpyHostToDevice);
    

    tracker_kalman_gain(&tracker1,2000);
    cudaDeviceSynchronize();

    float* K=(float*)malloc(sizeof(float)*n*m*N);
    cudaMemcpy(K,tracker1.d_K,sizeof(float)*n*m*N,cudaMemcpyDeviceToHost);
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(errora));
    }
    
    printf("output kalman gain K=P*H^T*S^(-1),  S=H*P*H^T+R:\n");
    for (int i=0;i<N;i++){
        if (abs(K[i*n*m]-0.952)>0.001 || abs(K[i*n*m+1]-0.238)>0.001){
            printf("error in K result, check");
            return;
        }
    }
    printf("No error.");
    tracker1.freeOnDevice();
    cudaFree(d_tracker);
    free(K);
    free(P);
}

int main(void){
    
    //test_predict_positions();
    //test_kalman_gain_single();
    //test_add_single();
    //test_transpose();
    //test_kalman_gain_batch();
    //test_MMaddKernel();
    test_tracker_kalman_gain();
    return 0;
}