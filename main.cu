#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hungarian_cpu_vectorized.h"
#include "helper.h"
#include "sort_lib.h"

int main(void){

    //set a max expected detection 2000
    int Max_Tracks=1000;
    int Max_detection = 1000;
    //set state variable and detection spec
    int m=4;     //detection
    int n=7;     //state variable

    tracker* d_tracker;
    cudaMalloc(&d_tracker,sizeof(tracker));
    tracker tracker1(Max_Tracks,Max_detection,m,n);
    tracker1.allocateOnDevice();
    tracker* tracker2 = (tracker*)malloc(sizeof(tracker));
    cudaMemcpy(d_tracker,&tracker1,sizeof(tracker),cudaMemcpyHostToDevice);
    cudaMemcpy(tracker2,d_tracker,sizeof(tracker),cudaMemcpyDeviceToHost);
    printf("max tracks: %d",tracker2->Max_Tracks);
    tracker1.freeOnDevice();
    cudaFree(d_tracker);
    free(tracker2);


    return 0;
}