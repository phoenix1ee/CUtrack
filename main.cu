#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/hungarian_cpu_vectorized.h"
#include "include/helper.h"
#include "include/sort_lib.h"


int main(void){

    //set a max expected detection 2000
    int Max_Tracks=1000;
    int Max_detection = 1000;
    //set state variable and detection spec
    int m=4;     //detection
    int n=7;     //state variable

    tracker tracker1(Max_Tracks,Max_detection,m,n);
    tracker1.allocateOnDevice();

    //setup input stream

    //setup inference model
    //fixed input and out tensor address

    //while true, get a frame
    // then start inference
    // 

    tracker1.freeOnDevice();
    return 0;
}