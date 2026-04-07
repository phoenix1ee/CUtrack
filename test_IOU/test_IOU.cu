#include <stdlib.h>
#include <stdio.h>
#include "../include/helper.h"
#include "../include/sort_lib.h"

int main(void){
    printf("testing IOU calculation with the self defined tracker object:\n");
    //set tracker variable
    int Max_track=2000;
    int Max_detection = 2000;
    //set state variable and detection spec
    int m=4;     //detection
    int n=7;     //state variable

    tracker tracker1(Max_track,Max_detection,m,n);
    tracker1.allocateOnDevice();

    int totaltrack = 16;
    std::cout<<"No. of tracks tested:"<<totaltrack<<", measurement vector size: "<<m
            <<", state vector size: "<<n<<", Using a 100*100 pixel map"<<"\n";
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

    std::cout<<"Input predicted state i and detection i is shifted by 1 pixel"
                <<"\neach predicted state i should have lowest cost at detection i\n";

    float*d_detectbox;
    cudaMalloc((void**)&d_detectbox,sizeof(float)*totaldetection*4);
    float* IOU=(float*)malloc(sizeof(float)*totaltrack*totaldetection);
    cudaMemcpy(tracker1.d_state_predicted,PS,sizeof(float)*totaltrack*7,cudaMemcpyHostToDevice);
    cudaMemcpy(d_detectbox,db,sizeof(float)*totaldetection*4,cudaMemcpyHostToDevice);
    dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid(1, 1, 1 );

    struct timespec start2;
	getstarttime(&start2);
    tracker_compute_IOU(&tracker1,d_detectbox,totaltrack,totaldetection,100,100);
    uint64_t consumed2 = get_lapsed(start2);
	printf("IOU kernel-used time: %" PRIu64 "\n",consumed2);
    cudaError_t errork = cudaGetLastError();
    if (errork != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(errork));
    }
    cudaMemcpy(IOU,tracker1.d_IOU,sizeof(float)*totaltrack*totaldetection,cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("ErrorP: %s\n", cudaGetErrorString(error));
    }
    tracker1.freeOnDevice();
    cudaFree(d_detectbox);
    free(PS);
    free(db);
    std::cout<<"tracks\\detection\n";
    printmatrix(IOU,totaldetection,totaltrack);
    free(IOU);
    return 0;
}