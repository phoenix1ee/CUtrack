#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include <cinttypes>
#include "include/hungarian_cpu_vectorized.h"
#include "include/sort_lib.h"

__global__ void dummykernel(void){}


void execute_cpu_functions(float* input, int width, int height){
	struct timespec start;
	getstarttime(&start);

	reduction_avx(input,width,height);
	uint64_t consumed = get_lapsed(start);
	printf("CPU used time: %" PRIu64 "\n",consumed);
	//printf("%  " PRIu64 , consumed);
}

void execute_gpu_functions(float* input, int totalsize, int blocksize, int width, int height){
	//make a copy 
	float* backup = (float*)malloc(sizeof(float)*totalsize);
	std::copy(input,input+totalsize,backup);

	dummykernel<<<1,1>>>();

	struct timespec start;
	getstarttime(&start);
	//run global memory version
	reductionglobalmem(input, totalsize, blocksize, width, height);
	uint64_t consumed = get_lapsed(start);
	printf("global memory-GPU used time: %" PRIu64 "\n",consumed);
	//printf(" % " PRIu64 , consumed);

	//copy back to input
	std::copy(backup,backup+totalsize,input);

	struct timespec start2;
	getstarttime(&start2);
	//run mapped memory version
	reductionmappedmem(input, totalsize, blocksize, width, height);
	uint64_t consumed2 = get_lapsed(start2);
	printf("mapped memory-GPU used time: %" PRIu64 "\n",consumed2);
	//printf(" % " PRIu64 , consumed2);

	//copy back to input
	std::copy(backup,backup+totalsize,input);

	struct timespec start3;
	getstarttime(&start3);
	//run streaming memory version
	reductionStreamMemory(input, totalsize, blocksize, width, height);
	uint64_t consumed3 = get_lapsed(start3);
	printf("stream memory-GPU used time: %" PRIu64 "\n",consumed3);
	//printf(" % " PRIu64 , consumed3);
	
	//copy back to input
	std::copy(backup,backup+totalsize,input);

	struct timespec start5;
	getstarttime(&start5);
	//run streaming memory version
	reductionNoTransposeStreamMemory(input, totalsize, blocksize, width, height);
	uint64_t consumed5 = get_lapsed(start5);
	printf("zero-transpose-kernels-stream mem-GPU used time: %" PRIu64 "\n",consumed5);
	//printf(" % " PRIu64 , consumed3);

	//copy back to input
	std::copy(backup,backup+totalsize,input);

	struct timespec start4;
	getstarttime(&start4);
	//run streaming memory version
	reductionNotranspose(input, totalsize, blocksize, width, height);
	uint64_t consumed4 = get_lapsed(start4);
	printf("zero-transpose-kernels-globalmem-GPU used time: %" PRIu64 "\n",consumed4);
	//printf(" % " PRIu64 , consumed3);

	free(backup);

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
	
	//printmatrix(input,width,height);
	//printf("\nafterprocessing\n");
	
	execute_cpu_functions(input2,width,height);
	execute_gpu_functions(input,totalThreads,blockSize,width,height);
	//printmatrix(input,width,height);
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
				printf("discrepancy found row: %d col: %d ",i,j);
				printf("cpu: %.2f, gpu: %.2f\n", input2[i*width+j],input[i*width+j]);
				break;
			}
		}
	}
	
	free(input2);
	free(input);
	return 0;
}