#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "helper.h"
#include <cinttypes>
#include "hungarian_cpu_vectorized.h"

__global__ void dummykernel(void){

}

__global__ void colreduction(float *input, int matrixwidth, int matrixheight){
	int blockoffset = (gridDim.x*blockIdx.y) + blockIdx.x;
	int gridsize = gridDim.x*gridDim.y;
	if (blockoffset >= matrixwidth) return;
	//a tile of 32*32
	__shared__ float sharedArray[32][33];
	__shared__ float sharedmin[32];
	//each blocks process 32 cols
	// assume blockDim.x=32
	int tid = threadIdx.y*32+threadIdx.x;
	if(tid<32){
		sharedmin[tid]=INFINITY;
	}
    int lane   = tid & 31;
	//loop thru each grid size across cols
	//each blocks process 32 cols
	for(int col=blockoffset*32;col<matrixwidth;col+=32*gridsize){
		//every 32 rows, a block read a tile of matrix to shared array
		for(int tileoffset=0;tileoffset<matrixheight;tileoffset+=32){
			//within each tile, a block iterate to read matrix and write to the sharedmem
			for(int row = threadIdx.y;row<32;row+=blockDim.y){
				int input_row = tileoffset+row;
				int input_col = col + threadIdx.x;
				if(input_row<matrixheight && input_col<matrixwidth){
					//within the matrix
					sharedArray[row][threadIdx.x]=input[input_row*matrixwidth+input_col];
				}
				else{
					//outside matrix, put inf to sharemem value
					sharedArray[row][threadIdx.x]=INFINITY;
				}
			}
			__syncthreads();
			//use the block to read through each tile
			for(int j=threadIdx.y;j<32;j+=blockDim.y){
				//within each tile, each warp in block read a column of shared array
				float localmin = sharedArray[threadIdx.x][j];
				//shuffle to found min of each warp
				for (int offset = 16; offset > 0; offset = offset /2) {
					localmin = min(localmin, __shfl_down_sync(0xffffffff, localmin, offset));
				}
				//at lane 0, check if value found is smaller than sharedmin and update
				if (lane == 0 && localmin<sharedmin[j]) {
					sharedmin[j] = localmin;
				}
			}
			__syncthreads();
		}
		__syncthreads();
		//after iterate through all rows, each col's min is at sharedmin[]
		for(int row=threadIdx.y;row<matrixheight;row+=blockDim.y){
			input[row*matrixwidth+col+threadIdx.x]-=sharedmin[threadIdx.x];
		}
		__syncthreads();
	}
	__syncthreads();
	return;
}
__global__ void rowreduction(float *a, int matrixwidth, int matrixheight){
    //row reduction
	int blockoffset = (gridDim.x*blockIdx.y) + blockIdx.x;
	int gridsize = gridDim.x*gridDim.y;
	int blocksize = blockDim.x*blockDim.y;
	if (blockoffset >= matrixheight) return;
	//support for blocksize up to 256*32, which is unlikely
	__shared__ float sharedmin[256];
	// assume blockDim.x=32
	int tid = threadIdx.y*32+threadIdx.x;
	int warpId = tid >> 5;
    int lane   = tid & 31;
	int numWarps = (blocksize + 31) / 32;
    for (int row = blockoffset; row < matrixheight; row += gridsize) {
		if(tid<256){
			sharedmin[tid]=INFINITY;
		}
		__syncthreads();
		//use each block as for a row
		int rowStart = row * matrixwidth;
		//find minimum
		float localMin = (float)INFINITY;
		// loop thru entire row by incremeenting w/ blocksize
		for (int x = tid; x < matrixwidth; x += blocksize) {
			//each thread in a block read 1 element
			localMin = min(localMin, a[rowStart+x]);
		}
		//find a minimum for each wrap
		for (int offset = 16; offset > 0; offset /= 2) {
			localMin = min(localMin, __shfl_down_sync(0xffffffff, localMin, offset));
		}
		if(lane==0){
			sharedmin[warpId]=localMin;
		}
		__syncthreads();
		// find the minimum from the shared mem on the warp 0
		if (warpId==0){
			float val = (lane < numWarps) ? sharedmin[lane] : INFINITY;
			for (int offset = 16; offset > 0; offset = offset /2) {
				val = min(val, __shfl_down_sync(0xffffffff, val, offset));
			}
			if (lane == 0) {
                sharedmin[0] = val;
            }
		}
		__syncthreads();
		float rowMin = sharedmin[0];
		__syncthreads();
		for (int x = tid; x < matrixwidth; x += blocksize) {
			//printf("%.2f ,%.2f ",a[rowStart+x] , rowMin);
			a[rowStart+x] =a[rowStart+x] - rowMin;
			//printf("%.2f , ",a[rowStart+x]);
        }
		__syncthreads();
    }
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

__global__ void transposeOnStream(float *a,float *b, int Mwidth, int Mheight, 
									int partitionID, int partitionHeight,int actualHeight){
	//do transpose
	//a: input matrix
	//b: output matrix

	//a tile of 32*32
	__shared__ float sharedArray[32][33];
	//each blocks process a tile
	//loop thru each grid size
	//(x,y) is the starting point of each block
	for(int y=blockIdx.y*32;y<actualHeight;y+=32*gridDim.y){
		for(int x=blockIdx.x*32;x<Mwidth;x+=32*gridDim.x){
				//do the processing
				//read a tile of matrix to shared array with multiple iterations
			for(int i = threadIdx.y;i<32;i+=blockDim.y){
				int aY = y + i;
            	int aX = x + threadIdx.x;
				if(aY<Mheight && aX<Mwidth){
					sharedArray[i][threadIdx.x]=a[aY*Mwidth+aX];
				}
			}
			__syncthreads();
			//read a column of shared array and write to a row of output array
			for(int j=threadIdx.y;j<32;j+=blockDim.y){
				int bY = x + j;
            	int bX = partitionID*partitionHeight + y + threadIdx.x;
				if(bY<Mwidth && bX<Mheight){
				b[bY*Mheight+bX]=sharedArray[threadIdx.x][j];
				}
			}
			//}
			__syncthreads();
		}
	}
	return;
}

__global__ void rowreductionOnStream(float *a,int strMwidth, int strMheight){
	//row reduction
	int blockoffset = (gridDim.x*blockIdx.y) + blockIdx.x;
	int gridsize = gridDim.x*gridDim.y;
	int blocksize = blockDim.x*blockDim.y;
	if (blockoffset >= strMheight) return;
	//support for blocksize up to 256*32, which is unlikely
	__shared__ float sharedmin[256];
	// assume blockDim.x=32
	int tid = threadIdx.y*32+threadIdx.x;
	int warpId = tid >> 5;
    int lane   = tid & 31;
	int numWarps = (blocksize + 31) / 32;
    for (int row = blockoffset; row < strMheight; row += gridsize) {
		if(tid<256){
			sharedmin[tid]=INFINITY;
		}
		__syncthreads();
		//use each block as for a row
		int rowStart = row * strMwidth;
		//find minimum
		float localMin = (float)INFINITY;
		// loop thru entire row by incremeenting w/ blocksize
		for (int x = tid; x < strMwidth; x += blocksize) {
			//each thread in a block read 1 element
			localMin = min(localMin, a[rowStart+x]);
		}
		//find a minimum for each wrap
		for (int offset = 16; offset > 0; offset /= 2) {
			localMin = min(localMin, __shfl_down_sync(0xffffffff, localMin, offset));
		}
		if(lane==0){
			sharedmin[warpId]=localMin;
		}
		__syncthreads();
		// find the minimum from the shared mem on the warp 0
		if (warpId==0){
			float val = (lane < numWarps) ? sharedmin[lane] : INFINITY;
			for (int offset = 16; offset > 0; offset = offset /2) {
				val = min(val, __shfl_down_sync(0xffffffff, val, offset));
			}
			if (lane == 0) {
                sharedmin[0] = val;
            }
		}
		__syncthreads();
		float rowMin = sharedmin[0];
		__syncthreads();
		for (int x = tid; x < strMwidth; x += blocksize) {
			//printf("%.2f ,%.2f ",a[rowStart+x] , rowMin);
			a[rowStart+x] =a[rowStart+x] - rowMin;
			//printf("%.2f , ",a[rowStart+x]);
        }
		__syncthreads();
    }
}

void reductionTransposestream1(float* input, float*d_input, float*d_temp, int totalThreads, int blocksize, int width, int height){
	//wrapper function for use streams and async memcpy for the reduction kernels

	//set a partition size to partition the array for reduction
	//use multiples of rows
	//say 32 rows per partitions
	int partitionheight = 32;
	int partitioncount=height/partitionheight+1;
	int partitionSize = width*partitionheight;
	cudaStream_t streamCopy, streamCompute, streamTranspose;
	cudaStreamCreate(&streamCopy);
	cudaStreamCreate(&streamCompute);
	cudaStreamCreate(&streamTranspose);

	//Use an array of events to track copying of all partitions
	cudaEvent_t* copyDone=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*partitioncount);

	//Use an array of events to track reduction of all partitions
	cudaEvent_t* Reduction1Done=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*partitioncount);


	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid(32, 1, 1 );

	for (int i = 0; i < partitioncount; i++) {
		//printf("get into loop");
		cudaEventCreate(&copyDone[i]);
		//printf("finishe creation event");
		// Calculate the pointer offset for this partition
		int offset = i * partitionSize;

		//check for the last partitions
		int currentSize = (i == partitioncount - 1) ? (totalThreads - offset) : partitionSize;
		int currentHeight = currentSize/width;
		// Start Async Copy in the Transfer Stream
		// Copy just 1 row each time
		cudaMemcpyAsync(d_input + offset, input + offset, 
						currentSize * sizeof(float), 
						cudaMemcpyHostToDevice, streamCopy);

		//printf("start memcpy");
		// Record an event in the Transfer Stream after copy
		cudaEventRecord(copyDone[i], streamCopy);

		//printf("mark record");
		// Make Compute Stream wait for THIS specific partition's event
		cudaStreamWaitEvent(streamCompute, copyDone[i], 0);

		cudaEventCreate(&Reduction1Done[i]);
		//printf("mark wait");
		// Launch the Kernel in the Compute Stream when copyDone[i] is signaled
		// Launch the kernel on only 1 partition of the data
		rowreductionOnStream<<<dimGrid, dimBlock, 0, streamCompute>>>(d_input+offset,width,currentHeight);
		
		cudaEventRecord(Reduction1Done[i], streamCompute);
		// Launch the Kernel in the Transpose Stream when Reduction1Done[i] is signaled
		// Launch the kernel on only 1 partition of the data
		cudaStreamWaitEvent(streamTranspose, Reduction1Done[i], 0);
		//launc the transpose
		transposeOnStream<<<dimGrid, dimBlock, 0, streamTranspose>>>(d_input+offset,d_temp,width,height,
																	i,partitionheight,currentHeight);
	}
	cudaStreamSynchronize(streamTranspose);

	cudaStreamDestroy(streamCopy);
	cudaStreamDestroy(streamCompute);
	cudaStreamDestroy(streamTranspose);

	free(copyDone);
	free(Reduction1Done);
}

void reductionTransposestream2(float* input, float*d_input, float*d_temp, int totalThreads, int blocksize, int width, int height){
	//wrapper function for use streams and async memcpy for the reduction kernels
	// this is the second part: second reduction, transpose back and memcpy back

	//set a partition size to partition the array for reduction
	//use multiples of rows
	//say 32 rows per partitions
	int partitionheight = 32;
	int partitioncount=height/partitionheight+1;
	int partitionSize = width*partitionheight;
	cudaStream_t streamCompute2, streamTranspose2;

	cudaStreamCreate(&streamCompute2);
	cudaStreamCreate(&streamTranspose2);

	//Use an array of events to track reduction of all partitions
	cudaEvent_t* Reduction2Done=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*partitioncount);

	//Use an array of events to track copying of all partitions
	cudaEvent_t* Transpose2Done=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*partitioncount);


	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid(32, 1, 1 );

	for (int i = 0; i < partitioncount; i++) {
		//printf("get into loop");
		//Create event for reduction
		cudaEventCreate(&Reduction2Done[i]);
		//printf("finishe creation event");
		// Calculate the pointer offset for this partition
		int offset = i * partitionSize;

		//check for the last partitions
		int currentSize = (i == partitioncount - 1) ? (totalThreads - offset) : partitionSize;
		int currentHeight = currentSize/width;
		// Launch the Kernel in the Compute Stream when copyDone[i] is signaled
		// Launch the kernel on only 1 partition of the data
		rowreductionOnStream<<<dimGrid, dimBlock, 0, streamCompute2>>>(d_input+offset,width,currentHeight);
		
		// Record an event in the StreamCompute2 after reduction done
		cudaEventRecord(Reduction2Done[i], streamCompute2);
		//printf("mark record");

		// Make Transpose Stream wait for THIS specific partition's event
		cudaStreamWaitEvent(streamTranspose2, Reduction2Done[i], 0);

		//Create event for transpose
		cudaEventCreate(&Transpose2Done[i]);

		// Launch the Kernel in the Transpose Stream 2 when Reduction2Done[i] is signaled
		// Launch the kernel on only 1 partition of the data
		transposeOnStream<<<dimGrid, dimBlock, 0, streamTranspose2>>>(d_input+offset,d_temp,width,height,
																	i,partitionheight,currentHeight);
		
		cudaEventRecord(Transpose2Done[i], streamTranspose2);
		
	}
	cudaStreamSynchronize(streamTranspose2);
	//cudaStreamSynchronize(streamCopy2);

	cudaStreamDestroy(streamCompute2);
	cudaStreamDestroy(streamTranspose2);
	
	free(Reduction2Done);
	free(Transpose2Done);
}

void reductionglobalmem(float* input, int totalThreads, int blocksize, int width, int height)
{
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel that using normal global memory
	//allocate device global memory for an input array and a buffer array of same size
	float* d_input;
	float* d_temp;

	cudaMalloc((void**)&d_input,sizeof(float)*totalThreads);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalThreads);
	cudaMemcpy(d_input,input,sizeof(float)*totalThreads,cudaMemcpyHostToDevice);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalThreads+blocksize-1)/blocksize, 1, 1 );
	rowreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();
	transpose<<<dimGrid,dimBlock>>>(d_input,d_temp,width,height);
	cudaDeviceSynchronize();
	//updatematrixsize(height,width);
	rowreduction<<<dimGrid,dimBlock>>>(d_temp,height,width);
	cudaDeviceSynchronize();
	transpose<<<dimGrid,dimBlock>>>(d_temp,d_input,height,width);
	cudaDeviceSynchronize();
	//updatematrixsize(width,height);
	
	cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf("kernel failed");
	}
	cudaMemcpy(input,d_input,sizeof(float)*totalThreads,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_input);
	cudaFree(d_temp);
	
	//printf("Global memory-GPU used time: %" PRIu64 "\n",consumed);
}

void reductionmappedmem(float* input, int totalThreads, int blocksize, int width, int height)
{
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel using pinned mapped memory

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//pin the allocate host memory
	cudaHostRegister(input,sizeof(float)*totalThreads,cudaHostRegisterMapped);
	//allocate pinned mapped memory for an input array and a buffer array of same size
	float* d_input;
	float* h_temp;
	float* d_temp;
	cudaError_t errora = cudaHostAlloc((void**)&h_temp,sizeof(float)*totalThreads,cudaHostAllocMapped);
	if (errora != cudaSuccess){
		printf("allcoation not success");
	}
	cudaHostGetDevicePointer((void**)&d_input,input,0);
	cudaHostGetDevicePointer((void**)&d_temp,h_temp,0);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalThreads+blocksize-1)/blocksize, 1, 1 );
	
	rowreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();
	transpose<<<dimGrid,dimBlock>>>(d_input,d_temp,width,height);
	cudaDeviceSynchronize();
	//updatematrixsize(height,width);
	rowreduction<<<dimGrid,dimBlock>>>(d_temp,height,width);
	cudaDeviceSynchronize();
	transpose<<<dimGrid,dimBlock>>>(d_temp,d_input,height,width);
	cudaDeviceSynchronize();
	//updatematrixsize(width,height);
	
	cudaError_t errorb = cudaGetLastError();
	if (errorb !=cudaSuccess){
		printf("mapped kernel failed");
	}
	cudaError_t errorc = cudaFreeHost(h_temp);
	if (errorc != cudaSuccess){
		printf("free not success");
	}
	cudaHostUnregister(input);

	//printf("Mapped memory-GPU used time: %" PRIu64 "\n",consumed);
}


void reductionStreamMemory(float* input, int totalThreads, int blocksize, int width, int height){
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel using pinned memory and async memcpy

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//pin the input host memory
	cudaHostRegister(input,sizeof(float)*totalThreads,cudaHostRegisterDefault);

	//allocate device global memory for an input array and a buffer array of same size
	float* d_input;
	float* d_temp;

	cudaMalloc((void**)&d_input,sizeof(float)*totalThreads);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalThreads);

	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalThreads+blocksize-1)/blocksize, 1, 1 );
	
	//call wrapper function
	reductionTransposestream1(input, d_input, d_temp, totalThreads, blocksize,width,height);
	cudaDeviceSynchronize();
	reductionTransposestream2(input, d_temp, d_input, totalThreads, blocksize,height,width);
	cudaDeviceSynchronize();

	cudaMemcpy(input,d_input,sizeof(float)*totalThreads,cudaMemcpyDeviceToHost);
	cudaHostUnregister(input);
	cudaFree(d_input);
	cudaFree(d_temp);

	//printf("stream memory-GPU used time: %" PRIu64 "\n",consumed);
}


void reductionNotranspose(float* input, int totalThreads, int blocksize, int width, int height)
{
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel that using normal global memory
	//allocate device global memory for an input array and a buffer array of same size
	float* d_input;
	float* d_temp;

	cudaMalloc((void**)&d_input,sizeof(float)*totalThreads);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalThreads);
	cudaMemcpy(d_input,input,sizeof(float)*totalThreads,cudaMemcpyHostToDevice);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalThreads+blocksize-1)/blocksize, 1, 1 );
	rowreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();
	//colreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();
	//updatematrixsize(width,height);
	
	cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf("kernel failed");
	}
	cudaMemcpy(input,d_input,sizeof(float)*totalThreads,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_input);
	cudaFree(d_temp);
	
	//printf("Global memory-GPU used time: %" PRIu64 "\n",consumed);
}

void execute_cpu_functions(float* input, int width, int height){
	struct timespec start;
	getstarttime(&start);

	reduction_avx(input,width,height);
	uint64_t consumed = get_lapsed(start);
	printf("CPU used time: %" PRIu64 "\n",consumed);
	//printf("%  " PRIu64 , consumed);
}

void execute_gpu_functions(float* input, int totalThreads, int blocksize, int width, int height){
	//make a copy 
	float* backup = (float*)malloc(sizeof(float)*totalThreads);
	std::copy(input,input+totalThreads,backup);

	dummykernel<<<1,1>>>();
	/*
	struct timespec start;
	getstarttime(&start);
	//run global memory version
	reductionglobalmem(input, totalThreads, blocksize, width, height);
	uint64_t consumed = get_lapsed(start);
	printf("global memory-GPU used time: %" PRIu64 "\n",consumed);
	//printf(" % " PRIu64 , consumed);

	//copy back to input
	std::copy(backup,backup+totalThreads,input);

	struct timespec start2;
	getstarttime(&start2);
	//run mapped memory version
	reductionmappedmem(input, totalThreads, blocksize, width, height);
	uint64_t consumed2 = get_lapsed(start2);
	printf("global memory-GPU used time: %" PRIu64 "\n",consumed2);
	//printf(" % " PRIu64 , consumed2);

	//copy back to input
	std::copy(backup,backup+totalThreads,input);

	struct timespec start3;
	getstarttime(&start3);
	//run streaming memory version
	reductionStreamMemory(input, totalThreads, blocksize, width, height);
	uint64_t consumed3 = get_lapsed(start3);
	printf("global memory-GPU used time: %" PRIu64 "\n",consumed3);
	//printf(" % " PRIu64 , consumed3);
	*/

	struct timespec start4;
	getstarttime(&start4);
	//run streaming memory version
	reductionNotranspose(input, totalThreads, blocksize, width, height);
	uint64_t consumed4 = get_lapsed(start4);
	printf("allinonekernel-globalmem-GPU used time: %" PRIu64 "\n",consumed4);
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
	//printf("matrix created for row and column reduction, row: %d, columns: %d\n",height,width);
	float* input = create_2d_array(width,height);
	//make a copy 
	float* input2 = (float*)malloc(sizeof(float)*totalThreads);
	std::copy(input,input+totalThreads,input2);
	
	printmatrix(input,width,height);
	printf("\nafterprocessing\n");
	
	execute_cpu_functions(input2,width,height);
	execute_gpu_functions(input,totalThreads,blockSize,width,height);
	printmatrix(input,width,height);
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