#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hungarian_cpu_vectorized.h"
#include "helper.h"
#include "sort_lib.h"

int main(void){
    int width=1<<10;
    int height=1<<10;
    float* matrix = create_2d_array(width,height);
	float* matrix2 = (float*)malloc(sizeof(float)*width*height);
	std::copy(matrix,matrix+width*height,matrix2);
	
	//printmatrix(input,width,height);
	//printf("\nafterprocessing\n");
	
	reduction_avx(matrix,width,height);
    reductionNotranspose(matrix2, width*height,256,width,height);
    //printmatrix(matrix,10,10);
    
    checkmatrix(matrix,matrix2,width*height);

    free(matrix);
    free(matrix2);
    
    return 0;
}