#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cinttypes>
#include <cuda_runtime.h>
#include "helper.h"
#include "sort_lib.h"
#include <device_launch_parameters.h>

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
}