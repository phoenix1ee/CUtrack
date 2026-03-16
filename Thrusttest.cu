#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "helper.h"
#include <cinttypes>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Create and return a pointer to an array of size rows and cols
// populate with random value
float* create_2d_array(int cols,int rows) {
    float* m = (float*)malloc(rows * cols * sizeof(float));
	srand(time(NULL));
    for (int i = 0; i < (rows * cols); i++) {
		// Initialize with random number between 1-999
        m[i] = rand() % 999+1;		
    }
    return m;
}


int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 24);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
	//create the array
	int width = (int)sqrt(totalThreads);
	int height = (int)sqrt(totalThreads);
	if (width*height!=totalThreads){
		while(totalThreads%width!=0){
			width +=1;
		}
		height = totalThreads / width;
	}
	printf("matrix created for row and column reduction, row: %d, columns: %d\n",height,width);
	float* input = create_2d_array(width,height);
	//make a copy 
	float* input2 = (float*)malloc(sizeof(float)*totalThreads);
	std::copy(input,input+totalThreads,input2);
	/*
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			//input[i*width+j]=(float)(i*width+j+1);
				printf("%.2f ",input[i*width+j]);
		}
		printf("\n");
	}
	printf("\nafterprocessing\n");
	*/
	//execute_cpu_functions(input2,width,height);
	//execute_gpu_functions(input,totalThreads,blockSize,width,height);
	/*
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
		
				printf("%.2f ",input[i*width+j]);
		}
		printf("\n");
	}
	*/
	/*
	//transposed
	for(int j=0;j<width;j++){
		for(int i=0;i<height;i++){
		
				printf("%.2f ",input[j*height+i]);
		}
		printf("\n");
	}
*/
	//validate cpu and gpu results
	
	
	//transpose stage validation
	/*
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			if (input2[i*width+j]!=input[j*height+i]){
				printf("discrepancy found row: %d col: %d",i,j);
				printf("cpu: %.2f, gpu: %.2f\n", input2[i*width+j],input[j*height+i]);
				break;
			}
		}
	}
	*/
	
	//final stage validation
	
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			if (input2[i*width+j]!=input[i*width+j]){
				printf("discrepancy found row: %d col: %d",i,j);
				printf("cpu: %.2f, gpu: %.2f\n", input2[i*width+j],input[i*width+j]);
				break;
			}
		}
	}
	
	free(input2);
	free(input);
	return 0;
}