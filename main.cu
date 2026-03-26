#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hungarian_cpu_vectorized.h"
#include "helper.h"
#include "sort_lib.h"

int main(void){

    int Max_Tracks=2000;
    //data structure and device correspondings
    //track IDs
    int* track_id=new int(Max_Tracks);
    int* d_track_id;
    //state variables
    float * state = new float(Max_Tracks*7);
    float * d_state;
    //state covariance matrix
    float * Pcov = new float(Max_Tracks*49);
    float*d_Pcov;
    //age of each tracks
    int* age = new int(Max_Tracks);
    int* d_age;
    //consecutive hit for each track
    int* hit_streak = new int(Max_Tracks);
    int* d_hit_streak;
    //consecutive hit for each track
    int* time_last_update = new int(Max_Tracks);
    int* d_time_last_update;
    // number of active tracks
    int count;
    

    //
    float* K=(float*)malloc(sizeof(float)*7*4*Max_Tracks);
    float* H=(float*)malloc(sizeof(float)*7*4);
    float* R=(float*)malloc(sizeof(float)*4*4);

    return 0;
}