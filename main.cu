#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hungarian_cpu_vectorized.h"
#include "helper.h"
#include "sort_lib.h"

int main(void){
    float* matrix = create_2d_array(10,10);
    printmatrix(matrix,10,10);
    printf("\n");

    reductionStreamMemory(matrix,100,256,10,10);
    printmatrix(matrix,10,10);
    free(matrix);
    return 0;
}