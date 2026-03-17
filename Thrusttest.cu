#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "helper.h"
#include "hungarian_cpu_vectorized.h"
#include <cinttypes>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>

void reductionfusion(float* input, int totalThreads, int blocksize, int width, int height){
	//do a stream cpy and row min reduction
	//sync
	//col reduction:
	//each block process 32 cols
	//each warp read 1 row
	//32 unit of shared mem
	//shuffle through each warp for each threadidx update shared mem[threadidx]
	//move down block for 32 rows
	//when done, do the reduction within the block
}

void reductionThrust(float* input, int totalelements, int blocksize, int cols, int rows){
		struct timespec start;
	getstarttime(&start);
	//wrap array for thrust and transfer to device
	thrust::device_vector<float> d_input_v(input, input + totalelements);

	//thrust::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
	/*
	1. keys_first, 2. keys_last: The input range of keys.
	3. values_first: The beginning of the input range of values, which must be the same length as the keys range.
	4. keys_output: The beginning of the output range for unique keys.
	5. values_output: The beginning of the output range for reduced values.

	The return value is a std::pair of iterators pointing to the end of the output key and value ranges, respectively. 
	*/
	// 1. Create keys: 0,0,0... 1,1,1... 2,2,2...
	auto key_it = thrust::make_transform_iterator(
		thrust::make_counting_iterator<int>(0),
		[cols] __host__ __device__ (int i) { return i / cols; }
	);

	// 2. Prepare output buffers
	thrust::device_vector<float> row_min(rows);

	// 3. Perform the reduction
	thrust::reduce_by_key(
		key_it, key_it + (rows * cols), // Keys (row indices)
		d_input_v.begin(),                // Values (the actual floats)
		thrust::make_discard_iterator(), // Don't waste VRAM storing row IDs
		row_min.begin(),
		thrust::equal_to<int>(),       // Binary predicate for keys
		thrust::minimum<float>()       // Reduction op
	);
	/*
	Assuming you have:

		d_data: Your original N×M array on the GPU.

		d_mins_out: The N minimum values you just calculated.

		// Use thrust::transform to process every single element in the original array
	*/
	thrust::transform(
		thrust::make_counting_iterator<int>(0),           // Global index (0 to N*M - 1)
		thrust::make_counting_iterator<int>(rows * cols), 
		d_input_v.begin(),                                   // Input: original values
		d_input_v.begin(),                                   // Output: overwritten values
		[cols, raw_mins = thrust::raw_pointer_cast(row_min.data())] __device__ (int i, float val) {
			// Calculate which row this index belongs to
			int row_idx = i / cols; 
			
			// Subtract the min of that specific row
			return val - raw_mins[row_idx];
		}
	);
	thrust::copy(d_input_v.begin(),d_input_v.end(),input);
	uint64_t consumed = get_lapsed(start);
	printf("Thrust GPU used time: %" PRIu64 "\n",consumed);
}

void execute_cpu_functions(float* input, int width, int height){
	struct timespec start;
	getstarttime(&start);
	//row reduction
	for(int i=0;i<height;i++){
		float rowmin = INFINITY;
		for(int j=0;j<width;j++){
			rowmin = (input[i*width+j]<rowmin)?input[i*width+j]:rowmin;
		}
		for(int j=0;j<width;j++){
			input[i*width+j]-=rowmin;
		}
	}
	/*
	//column reduction
	for(int j=0;j<width;j++){
		float colmin = INFINITY;
		for(int i=0;i<height;i++){
			colmin = (input[i*width+j]<colmin)?input[i*width+j]:colmin;
		}
		for(int i=0;i<height;i++){
			input[i*width+j]-=colmin;
		}
	}
	*/
	uint64_t consumed = get_lapsed(start);
	printf("CPU used time: %" PRIu64 "\n",consumed);
}

void execute_gpu_functions(float* input, int totalThreads, int blocksize, int width, int height){
	//run global memory version
	reductionThrust(input, totalThreads, blocksize, width, height);
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
	printmatrix(input,width,height);
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