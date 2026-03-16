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

void reductionThrust(float* input, int totalThreads, int blocksize, int width, int height){
	//wrap array for thrust and transfer to device
	thrust::device_vector<float> h_input_v(input, input + totalThreads);

	//thrust::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
	/*
	1. keys_first, 2. keys_last: The input range of keys.
	3. values_first: The beginning of the input range of values, which must be the same length as the keys range.
	4. keys_output: The beginning of the output range for unique keys.
	5. values_output: The beginning of the output range for reduced values.

	The return value is a std::pair of iterators pointing to the end of the output key and value ranges, respectively. 
	// 1. Create keys: 0,0,0... 1,1,1... 2,2,2...
	auto key_it = thrust::make_transform_iterator(
		thrust::make_counting_iterator<int>(0),
		[cols] __host__ __device__ (int i) { return i / cols; }
	);

	// 2. Prepare output buffers
	thrust::device_vector<int> d_keys_out(rows);
	thrust::device_vector<float> d_mins_out(rows);

	// 3. Perform the reduction
	thrust::reduce_by_key(
		key_it, key_it + (rows * cols), // Keys (row indices)
		d_data.begin(),                // Values (the actual floats)
		d_keys_out.begin(),
		d_mins_out.begin(),
		thrust::equal_to<int>(),       // Binary predicate for keys
		thrust::minimum<float>()       // Reduction op
	);

		// Use thrust::transform to process every single element in the original array
	thrust::transform(
		thrust::make_counting_iterator<int>(0),           // Global index (0 to N*M - 1)
		thrust::make_counting_iterator<int>(rows * cols), 
		d_data.begin(),                                   // Input: original values
		d_data.begin(),                                   // Output: overwritten values
		[cols, raw_mins = thrust::raw_pointer_cast(d_mins_out.data())] __device__ (int i, float val) {
			// Calculate which row this index belongs to
			int row_idx = i / cols; 
			
			// Subtract the min of that specific row
			return val - raw_mins[row_idx];
		}
	);
	
	*/
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