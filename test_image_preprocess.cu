#include "include/sort_lib.h"
#include "include/helper.h"
__global__ void dummykernel(void){}
void printmatrix_uint8(uint8_t*input,int cols,int rows){
    for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
		
				printf("%2d ",input[i*cols+j]);
		}
		printf("\n");
	}
}

void test_BGRconversion(void){
    int width = 1000;
    int height = 1000;
    uint8_t* input = (uint8_t*)malloc(sizeof(uint8_t)*width*height*3);
    for(int i=0;i<width*height;i++){
        input[i*3]=3;
        input[i*3+1]=2;
        input[i*3+2]=1;        
    }
    printmatrix_uint8(input,3,3);
    printmatrix_uint8(input+3*height*width-9,3,3);
    uint8_t* d_input;
    uint8_t* d_output;
    cudaMalloc((void**)&d_input,sizeof(uint8_t)*width*height*3);
    cudaMalloc((void**)&d_output,sizeof(uint8_t)*width*height*3);
    cudaMemcpy(d_input,input,sizeof(uint8_t)*width*height*3,cudaMemcpyHostToDevice);


    dummykernel<<<1,1>>>();
    struct timespec start2;
    getstarttime(&start2);
    frame_BGRtoRGB(d_input,width*height);
    uint64_t consumed2 = get_lapsed(start2);
	printf("simple mem BGR conversion-used time: %" PRIu64 "\n",consumed2);
    
    cudaMemcpy(input,d_input,sizeof(uint8_t)*width*height*3,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    printmatrix_uint8(input,3,3);
    printmatrix_uint8(input+3*height*width-9,3,3);

    for(int i=0;i<width*height;i++){
        if(input[i*3]!=1 || input[i*3+1]!=2 ||  input[i*3+2]!=3){
            printf("operation is wrong\n");
            return;
        }
    }
    free(input);
}

void test_preprocess(void){
    int width = 1280;
    int height = 720;
    uint8_t* input = (uint8_t*)malloc(sizeof(uint8_t)*width*height*3);
    for(int i=0;i<width*height;i++){
        input[i*3]=120;
        input[i*3+1]=120;
        input[i*3+2]=120;        
    }

    int out_width = 640;
    int out_height = 640;
    

    float* output = (float*)malloc(sizeof(float)*out_height*out_width*3);
    for(int i=0;i<out_height*out_width*3;i++){
        output[i]=1.0f;
    }

    uint8_t* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input,sizeof(uint8_t)*width*height*3);
    cudaMalloc((void**)&d_output,sizeof(float)*out_height*out_width*3);
    cudaMemcpy(d_input,input,sizeof(uint8_t)*width*height*3,cudaMemcpyHostToDevice);
    cudaMemcpy(d_output,output,sizeof(float)*out_height*out_width*3,cudaMemcpyHostToDevice);
    
    dummykernel<<<1,1>>>();
	cudaDeviceSynchronize();
    struct timespec start2;
    getstarttime(&start2);
    
    frame_preprocess(d_input,d_output,height,width,out_height,out_width);
    
    uint64_t consumed2 = get_lapsed(start2);
	printf("preprocess conversion-used time: %" PRIu64 "\n",consumed2);
    
    cudaMemcpy(output,d_output,sizeof(float)*out_height*out_width*3,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    free(input);
    free(output);
}

int main(void){
    test_BGRconversion();
    test_preprocess();
}