/**
 * @file hungarian_lib.cu
 * @author SF Lee
 * @date 2026-04-07
 * @brief GPU-accelerated implementation of the Hungarian algorithm.
 *
 * This file contains CUDA kernels and wrapper functions
 * used to compute optimal assignments using the Hungarian method.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cinttypes>
#include <cuda_runtime.h>
#include "include/helper.h"
#include "include/sort_lib.h"


__global__ void colreduction(float *input, int matrixwidth, int matrixheight){
	//blockoffset = starting col of each block
	int blockoffset = (gridDim.x*blockIdx.y) + blockIdx.x;
	int gridsize = gridDim.x*gridDim.y;
	if (blockoffset >= matrixwidth) return;
	//a tile of 32*32
	__shared__ float sharedArray[32][33];
	//32 min for 32 cols
	__shared__ float sharedmin[33];
	//each blocks process 32 cols
	//assume block width/blockDim.x=32
	int tid = threadIdx.y*32+threadIdx.x;
	//loop thru each grid size across cols
	//each blocks process 32 cols
	for(int col=blockoffset*32;col<matrixwidth;col+=32*gridsize){
		if(tid<32){
			sharedmin[tid]=(float)INFINITY;
		}
		//every 32 rows, a block read a tile of matrix to shared array
		for(int tileoffset=0;tileoffset<matrixheight;tileoffset+=32){
			//within each tile, a block iterate downward to read a total of 32 rows of the tile and write to the sharedmem
			for(int tile_row = threadIdx.y;tile_row<32;tile_row+=blockDim.y){
				int input_row = tileoffset+tile_row;
				int input_col = col + threadIdx.x;
				if(input_row<matrixheight && input_col<matrixwidth){
					//within the matrix
					sharedArray[tile_row][threadIdx.x]=input[input_row*matrixwidth+input_col];
				}
				else{
					//outside matrix, put inf to sharemem value
					sharedArray[tile_row][threadIdx.x]=(float)INFINITY;
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
				if (threadIdx.x == 0 && localmin<sharedmin[j]) {
					sharedmin[j] = localmin;
				}
			}

		}
		__syncthreads();
		//after iterate through all rows, each col's min is at sharedmin[]
		for(int row=threadIdx.y;row<matrixheight;row+=blockDim.y){
			if (col+threadIdx.x<matrixwidth){
				input[row*matrixwidth+col+threadIdx.x]-=sharedmin[threadIdx.x];
			}
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

void memcpyAndrowreductionstream_v2(float* input, float*d_input, int totalsize, int blocksize, int width, int height){
	//wrapper function for use streams and async memcpy for the reduction kernels

	//set a partition size to partition the array for reduction
	//use multiples of rows
	//say 32 rows per partitions
	int partitionheight = 32;
	int partitioncount=height/partitionheight+1;
	int partitionSize = width*partitionheight;
	cudaStream_t streamCopy, streamCompute;
	cudaStreamCreate(&streamCopy);
	cudaStreamCreate(&streamCompute);

	//Use an array of events to track copying of all partitions
	cudaEvent_t odd_copy;

	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid(32, 1, 1 );

	for (int i = 0; i < partitioncount; i++) {
		//printf("get into loop");
		cudaEventCreate(&odd_copy);
		//printf("finishe creation event");
		// Calculate the pointer offset for this partition
		int offset = i * partitionSize;

		//check for the last partitions
		int currentSize = (i == partitioncount - 1) ? (totalsize - offset) : partitionSize;
		int currentHeight = currentSize/width;
		// Start Async Copy in the Transfer Stream
		// Copy just 1 row each time
		cudaMemcpyAsync(d_input + offset, input + offset, 
						currentSize * sizeof(float), 
						cudaMemcpyHostToDevice, streamCopy);

		//printf("start memcpy");
		// Record an event in the Transfer Stream after copy
		cudaEventRecord(odd_copy, streamCopy);
		//printf("mark record");
		// Make Compute Stream wait for THIS specific partition's event
		cudaStreamWaitEvent(streamCompute, odd_copy, 0);
		cudaEventSynchronize(odd_copy);
		//printf("mark wait");
		// Launch the Kernel in the Compute Stream when copyDone[i] is signaled
		// Launch the kernel on only 1 partition of the data
		rowreductionOnStream<<<dimGrid, dimBlock, 0, streamCompute>>>(d_input+offset,width,currentHeight);

	}
	
	cudaStreamSynchronize(streamCompute);
	cudaStreamDestroy(streamCopy);
	cudaStreamDestroy(streamCompute);


}

void memcpyAndrowreductionstream(float* input, float*d_input, int totalsize, int blocksize, int width, int height){
	//wrapper function for use streams and async memcpy for the reduction kernels

	//set a partition size to partition the array for reduction
	//use multiples of rows
	//say 32 rows per partitions
	int partitionheight = 32;
	int partitioncount=height/partitionheight+1;
	int partitionSize = width*partitionheight;
	cudaStream_t streamCopy, streamCompute;
	cudaStreamCreate(&streamCopy);
	cudaStreamCreate(&streamCompute);

	//Use an array of events to track copying of all partitions
	cudaEvent_t* copyDone=(cudaEvent_t*)malloc(sizeof(cudaEvent_t)*partitioncount);

	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid(32, 1, 1 );

	for (int i = 0; i < partitioncount; i++) {
		//printf("get into loop");
		cudaEventCreate(&copyDone[i]);
		//printf("finishe creation event");
		// Calculate the pointer offset for this partition
		int offset = i * partitionSize;

		//check for the last partitions
		int currentSize = (i == partitioncount - 1) ? (totalsize - offset) : partitionSize;
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

		//printf("mark wait");
		// Launch the Kernel in the Compute Stream when copyDone[i] is signaled
		// Launch the kernel on only 1 partition of the data
		rowreductionOnStream<<<dimGrid, dimBlock, 0, streamCompute>>>(d_input+offset,width,currentHeight);

	}
	cudaStreamSynchronize(streamCompute);
	cudaStreamDestroy(streamCopy);
	cudaStreamDestroy(streamCompute);

	free(copyDone);
}

void memcpy_reduction_Transpose_stream(float* input, float*d_input, float*d_temp, int totalsize, int blocksize, int width, int height){
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
		int currentSize = (i == partitioncount - 1) ? (totalsize - offset) : partitionSize;
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

void reduction_Transpose_stream(float* input, float*d_input, float*d_temp, int totalsize, int blocksize, int width, int height){
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
		int currentSize = (i == partitioncount - 1) ? (totalsize - offset) : partitionSize;
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

void reductionglobalmem(float* input, int totalsize, int blocksize, int width, int height){
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel that using normal global memory
	//allocate device global memory for an input array and a buffer array of same size
	float* d_input;
	float* d_temp;

	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalsize);
	cudaMemcpy(d_input,input,sizeof(float)*totalsize,cudaMemcpyHostToDevice);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalsize+blocksize-1)/blocksize, 1, 1 );
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
		printf("kernel failed-globalmem\n");
	}
	cudaMemcpy(input,d_input,sizeof(float)*totalsize,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_input);
	cudaFree(d_temp);

}

void reductionmappedmem(float* input, int totalsize, int blocksize, int width, int height){
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel using pinned mapped memory

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//pin the allocate host memory
	cudaHostRegister(input,sizeof(float)*totalsize,cudaHostRegisterMapped);
	//allocate pinned mapped memory for an input array and a buffer array of same size
	float* d_input;
	float* h_temp;
	float* d_temp;
	cudaError_t errora = cudaHostAlloc((void**)&h_temp,sizeof(float)*totalsize,cudaHostAllocMapped);
	if (errora != cudaSuccess){
		printf("allcoation not success");
	}
	cudaHostGetDevicePointer((void**)&d_input,input,0);
	cudaHostGetDevicePointer((void**)&d_temp,h_temp,0);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalsize+blocksize-1)/blocksize, 1, 1 );
	
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

void reductionStreamMemory(float* input, int totalsize, int blocksize, int width, int height){
	//update the constant value
	//updatematrixsize(width,height);
	//launch kernel using pinned memory and async memcpy

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//pin the input host memory
	cudaHostRegister(input,sizeof(float)*totalsize,cudaHostRegisterDefault);

	//allocate device global memory for an input array and a buffer array of same size
	float* d_input;
	float* d_temp;

	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);
	cudaMalloc((void**)&d_temp,sizeof(float)*totalsize);

	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalsize+blocksize-1)/blocksize, 1, 1 );
	
	//call wrapper function
	memcpy_reduction_Transpose_stream(input, d_input, d_temp, totalsize, blocksize,width,height);
	cudaDeviceSynchronize();
	reduction_Transpose_stream(input, d_temp, d_input, totalsize, blocksize,height,width);
	cudaDeviceSynchronize();

	cudaError_t errorb = cudaGetLastError();
	if (errorb !=cudaSuccess){
		printf("stream mem kernel failed");
	}
	cudaMemcpy(input,d_input,sizeof(float)*totalsize,cudaMemcpyDeviceToHost);
	cudaHostUnregister(input);
	cudaFree(d_input);
	cudaFree(d_temp);
}

void reductionNoTransposeStreamMemory(float* input, int totalsize, int blocksize, int width, int height){

	//launch kernel using pinned memory and async memcpy, and no transpose col reduction

	cudaSetDeviceFlags(cudaDeviceMapHost);
	//pin the input host memory
	cudaHostRegister(input,sizeof(float)*totalsize,cudaHostRegisterDefault);

	//allocate device global memory for an input array
	float* d_input;

	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);

	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalsize+blocksize-1)/blocksize, 1, 1 );
	
	//call wrapper function
	memcpyAndrowreductionstream(input, d_input, totalsize, blocksize,width,height);
	
	cudaDeviceSynchronize();
	colreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf("no transpose stream kernel failed");
	}
	cudaDeviceSynchronize();

	cudaMemcpy(input,d_input,sizeof(float)*totalsize,cudaMemcpyDeviceToHost);
	cudaHostUnregister(input);
	cudaFree(d_input);

}

void reductionNotranspose(float* input, int totalsize, int blocksize, int width, int height){
	//launch kernel that using normal global memory and no transpose col reduction
	//allocate device global memory for an input array
	float* d_input;

	cudaMalloc((void**)&d_input,sizeof(float)*totalsize);
	cudaMemcpy(d_input,input,sizeof(float)*totalsize,cudaMemcpyHostToDevice);
	//define grid and block size
	dim3 dimBlock(32, blocksize/32, 1 );
	dim3 dimGrid((totalsize+blocksize-1)/blocksize, 1, 1 );

	rowreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();

	colreduction<<<dimGrid,dimBlock>>>(d_input,width,height);
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf("no transpose kernel failed");
	}
	cudaDeviceSynchronize();
	cudaMemcpy(input,d_input,sizeof(float)*totalsize,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_input);
	
}

void transposeArray(float *d_a,float *d_b,int matrixwidth, int matrixheight){
	//define grid and block size
	dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((matrixheight*matrixwidth+255)/256, 1, 1 );
	transpose<<<dimGrid,dimBlock>>>(d_a,d_b,matrixwidth,matrixheight);
	cudaDeviceSynchronize();
}