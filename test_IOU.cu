#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
#include "sort_lib.h"

int main(void){
    int totaltrack = 16;
    float*PS=(float*)malloc(sizeof(float)*totaltrack*7);
    for(int i=0;i<totaltrack;i++){
        PS[i*7]=rand()%100;
        PS[i*7+1]=rand()%100;
        PS[i*7+2]=16;
        PS[i*7+3]=1;
        PS[i*7+4]=1;
        PS[i*7+5]=1;
        PS[i*7+6]=1;
    }
    int totaldetection = 16;
    float*db=(float*)malloc(sizeof(float)*totaldetection*4);
    for(int i=0;i<totaldetection;i++){
        db[i*4]=(PS[i*7]+1)/100;
        db[i*4+1]=(PS[i*7+1]+1)/100;
        db[i*4+2]=4;
        db[i*4+3]=4;
    }

    float* IOU=(float*)malloc(sizeof(float)*totaltrack*totaldetection);
    float * d_p;
    float * d_d;
    float * IOUM;
    cudaMalloc((void**)&d_p,sizeof(float)*totaltrack*7);
    cudaMalloc((void**)&d_d,sizeof(float)*totaldetection*4);
    cudaMemcpy(d_p,PS,sizeof(float)*totaltrack*7,cudaMemcpyHostToDevice);
    cudaMemcpy(d_d,db,sizeof(float)*totaldetection*4,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&IOUM,sizeof(float)*totaltrack*totaldetection);
    dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid(1, 1, 1 );

    struct timespec start2;
	getstarttime(&start2);
    computeIOUmatrix<<<dimGrid,dimBlock>>>(d_p,d_d,IOUM,totaltrack,totaldetection,100,100);
    uint64_t consumed2 = get_lapsed(start2);
	printf("IOU kernel-used time: %" PRIu64 "\n",consumed2);
    cudaError_t errork = cudaGetLastError();
    if (errork != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errork));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(IOU,IOUM,sizeof(float)*totaltrack*totaldetection,cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(error));
    }
    cudaFree(d_d);
    cudaFree(d_p);
    cudaFree(IOUM);
    free(PS);
    free(db);
    printmatrix(IOU,totaldetection,totaltrack);
    free(IOU);
    return 0;
}