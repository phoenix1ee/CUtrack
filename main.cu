#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hungarian_cpu_vectorized.h"
#include "helper.h"
#include "sort_lib.h"

__global__ void trackertestinitialize(tracker* trackA){
    //fill in the track 0 and track N-1for testing
    *trackA->d_activetracks=2;
    *trackA->d_currentdetections=2;
    trackA->d_track_id[0]=12345;
    for(int i=0;i<trackA->n;i++){
        trackA->d_state[i]=0.9;
    }

    for(int i=0;i<trackA->n*trackA->n;i++){
        trackA->d_Pcov[i]=0.1;
    }

    trackA->d_age[0]=10;

    trackA->d_hit_streak[0]=5;

    for(int i=0;i<trackA->m;i++){
        trackA->d_Z[i]=0.2;
    }

    for(int i=0;i<trackA->m*trackA->m;i++){
        trackA->d_S[i]=0.3;
    }

    for(int i=0;i<trackA->m*trackA->n;i++){
        trackA->d_K[i]=0.4;
    }

    for(int i=0;i<trackA->n*trackA->m;i++){
        trackA->d_H[i]=0.5;
    }
    for(int i=0;i<trackA->n*trackA->m;i++){
        trackA->d_R[i]=0.6;
    }

    trackA->d_track_id[trackA->Max_Tracks-1]=54321;
    for(int i=0;i<trackA->n;i++){
        trackA->d_state[(trackA->Max_Tracks-1)*trackA->n+i]=0.99;
    }
    for(int i=0;i<trackA->n*trackA->n;i++){
        trackA->d_Pcov[(trackA->Max_Tracks-1)*trackA->n*trackA->n+i]=0.11;
    }
    trackA->d_age[trackA->Max_Tracks-1]=11;
    trackA->d_hit_streak[trackA->Max_Tracks-1]=7;
    
    for(int i=0;i<trackA->m;i++){
        trackA->d_Z[(trackA->Max_detection-1)*trackA->m+i]=0.22;
    }
    
    for(int i=0;i<trackA->m*trackA->m;i++){
        trackA->d_S[(trackA->Max_Tracks-1)*trackA->m*trackA->m+i]=0.33;
    }
    for(int i=0;i<trackA->m*trackA->n;i++){
        trackA->d_K[(trackA->Max_Tracks-1)*trackA->m*trackA->n+i]=0.44;
    }

}

__global__ void trackertestinitializeprint(tracker* trackA){
    //fill in the track 0 and track N-1for testing
    printf("track id 0: %d and last %d \n",trackA->d_track_id[0],trackA->d_track_id[trackA->Max_Tracks-1]);
    
    printf("d_state 0:\n");
    for(int i=0;i<trackA->n;i++){
        printf("%.2f ",trackA->d_state[i]);
    }
    printf("\n");

    printf("d_state last:\n");
    for(int i=0;i<trackA->n;i++){
        printf("%.2f ",trackA->d_state[(trackA->Max_Tracks-1)*trackA->n+i]);
    }
    printf("\n");

    printf("covariance matrix 0:\n");
    printmatrix_colmajor_ondevice(trackA->d_Pcov,trackA->n,trackA->n);
    printf("covariance matrix N:\n");
    printmatrix_colmajor_ondevice(trackA->d_Pcov+(trackA->Max_Tracks-1)*trackA->n*trackA->n,trackA->n,trackA->n);

    printf("age 0 %d\n",trackA->d_age[0]);
    printf("age N %d\n",trackA->d_age[trackA->Max_Tracks-1]);

    printf("hit streak 0 %d\n",trackA->d_hit_streak[0]);
    printf("hit streak N %d\n",trackA->d_hit_streak[trackA->Max_Tracks-1]);

    printf("no of active tracks %d \n",*trackA->d_activetracks);

    printf("d_Z 0:\n");
    for(int i=0;i<trackA->m;i++){
        printf("%.2f ",trackA->d_Z[i]);
    }
    printf("\n");

    printf("d_Z last:\n");
    for(int i=0;i<trackA->m;i++){
        printf("%.2f ",trackA->d_Z[(trackA->Max_detection-1)*trackA->m+i]);
    }
    printf("\n");

    printf("no of current detection %d\n",*trackA->d_currentdetections);

    printf("d_S 0:\n");
    printmatrix_colmajor_ondevice(trackA->d_S,trackA->m,trackA->m);
    printf("d_S last:\n");
    printmatrix_colmajor_ondevice(trackA->d_S+(trackA->Max_Tracks-1)*trackA->m*trackA->m,trackA->m,trackA->m);
    
    printf("d_K 0:\n");
    printmatrix_colmajor_ondevice(trackA->d_K,trackA->n,trackA->m);
    printf("d_K last:\n");
    printmatrix_colmajor_ondevice(trackA->d_K+(trackA->Max_Tracks-1)*trackA->m*trackA->n,trackA->n,trackA->m);

    printf("d_H last:\n");
    printmatrix_colmajor_ondevice(trackA->d_H,trackA->m,trackA->n);
    printf("d_R last:\n");
    printmatrix_colmajor_ondevice(trackA->d_R,trackA->m,trackA->m);
    
}

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
    cudaMemcpy(d_tracker,&tracker1,sizeof(tracker),cudaMemcpyHostToDevice);
    trackertestinitialize<<<1,1>>>(d_tracker);
    cudaDeviceSynchronize();
    trackertestinitializeprint<<<1,1>>>(d_tracker);

    tracker1.freeOnDevice();
    cudaFree(d_tracker);

    return 0;
}