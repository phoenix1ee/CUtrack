#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.h"
#include <cinttypes>


__global__ void dummykernel(void){

}

__global__ void transpose(float *a,float *b,int matrixwidth, int matrixheight){
	//do transpose

	//a tile of 32*32
	__shared__ float sharedArray[32][33];
	//each blocks process a tile
	//loop thru each grid size
	//(x,y) is the starting point of each block
	for(int y=blockIdx.y*32;y<matrixheight;y+=32*gridDim.y){
		for(int x=blockIdx.x*32;x<matrixwidth;x+=32*gridDim.x){
				//do the processing
				//read a tile of matrix to shared array with multiple iterations
			for(int i = threadIdx.y;i<32;i+=blockDim.y){
				int aY = y + i;
            	int aX = x + threadIdx.x;
				if(aY<matrixheight && aX<matrixwidth){
					sharedArray[i][threadIdx.x]=a[aY*matrixwidth+aX];
				}
			}
			__syncthreads();
			//read a column of shared array and write to a row of output array
			for(int j=threadIdx.y;j<32;j+=blockDim.y){
				int bY = x + j;
            	int bX = y + threadIdx.x;
				if(bY<matrixwidth && bX<matrixheight){
				b[bY*matrixheight+bX]=sharedArray[threadIdx.x][j];
				}
			}
			//}
			__syncthreads();
		}
	}
	return;
}

void mytranspose(float* d_input, float* d_temp, int height, int width){
	dim3 dimBlock(32, 256, 1 );
	dim3 dimGrid((height*width+256-1)/256, 1, 1 );
	transpose<<<dimGrid,dimBlock>>>(d_input,d_temp,width,height);
	cudaDeviceSynchronize();
}

void cublasTranspose_simple(float* d_in, float* d_out, int o_row, int o_col){
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t T_handle;
    cublasCreate(&T_handle);

    cublasStatus_t T_status=CUBLAS_STATUS_SUCCESS;

    //transpose the matrix from o_row*o_col to o_col to o_row
    cublasSgeam(T_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                o_col,o_row,
                &alpha,
                d_in, o_row,
                &beta,
                nullptr, o_col,
                d_out, o_col);

    cudaDeviceSynchronize();
    if (T_status!=CUBLAS_STATUS_SUCCESS){
        printf("cublas geam to transpose failed");
    }
    cublasDestroy(T_handle);
}

void mytransposetest(float* input, int totalsize, int blocksize, int width, int height){
	float* d_input;
	float* d_temp;
	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalsize);
	cudaMemcpy(d_input,input,sizeof(float)*totalsize,cudaMemcpyHostToDevice);
		struct timespec start;
	getstarttime(&start);
	mytranspose(input,d_temp,height,width);
		uint64_t consumed = get_lapsed(start);
	printf("my transpose memory-GPU used time: %" PRIu64 "\n",consumed);
	cudaFree(d_temp);
	cudaFree(d_input);
}
void cublastransposetest(float* input, int totalsize, int blocksize, int width, int height){
	float* d_input;
	float* d_temp;
	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalsize);
	cudaMemcpy(d_input,input,sizeof(float)*totalsize,cudaMemcpyHostToDevice);
		struct timespec start;
	getstarttime(&start);
	cublasTranspose_simple(input,d_temp,height,width);
			uint64_t consumed = get_lapsed(start);
	printf("cublas memory-GPU used time: %" PRIu64 "\n",consumed);
	cudaFree(d_temp);
	cudaFree(d_input);

}

void execute_cpu_functions(float* input, int width, int height){

}

void execute_gpu_functions(float* input, int totalsize, int blocksize, int width, int height){
	//make a copy 
	float* backup = (float*)malloc(sizeof(float)*totalsize);
	std::copy(input,input+totalsize,backup);

	dummykernel<<<1,1>>>();

	mytransposetest(input, totalsize, blocksize, width, height);

	//copy back to input
	std::copy(backup,backup+totalsize,input);

	//run mapped memory version
	cublastransposetest(input, totalsize, blocksize, width, height);
	//printf(" % " PRIu64 , consumed2);

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