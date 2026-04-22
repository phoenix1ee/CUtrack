#include "include/sort_lib.h"
#include "include/helper.h"

__global__ void preprocess(uint8_t* d_frame_in,float* d_frame_out,
                        int h_in, int w_in, int out_h, int out_w,
                        int padx, int pady, float scale){
    //given a frame in BGR HWC mode, resize frame to given size and 
    //convert to RGB CHW with normalized RGB value
    int y_offset=blockIdx.y*blockDim.y;
    
    int x_offset=blockIdx.x*blockDim.x;

    //each thread handle 1 pixel
    for(int y=y_offset;y<out_h;y+=gridDim.y*blockDim.y){
		for(int x=x_offset;x<out_w;x+=gridDim.x*blockDim.x){
            int pixel_x=x+threadIdx.x;
            int pixel_y=y+threadIdx.y;
            int HW = out_w * out_h;
            int pixel_id = pixel_y * out_w + pixel_x;

            int pixelbyte_r= pixel_id;
            int pixelbyte_g= HW+pixel_id;
            int pixelbyte_b= HW*2+pixel_id;

            if(pixel_x<padx || pixel_x>=out_w-padx || pixel_y<pady || pixel_y>=out_h-pady){
                //in blank area
                d_frame_out[pixelbyte_r]=0.0f;
                d_frame_out[pixelbyte_g]=0.0f;
                d_frame_out[pixelbyte_b]=0.0f;
            }
            else{
                // Map to input coordinates
                float x_in = (pixel_x - padx) / scale;
                float y_in = (pixel_y - pady) / scale;
                // find the nearest 4 pixel in original frame
                // use bilinear interpolation to find new pixel values
                int x0 = floor(x_in);
                int y0 = floor(y_in);
                int x1 = min(x0 + 1, w_in - 1);
                int y1 = min(y0 + 1, h_in - 1);
                //x0+i+j=x1
                //y0+s+t=y1
                float i = x_in-x0;
                float j = 1-i;
                float s = y_in-y0;
                float t = 1-s;

                //read the bgr from frame
                uchar3 x0y0 = make_uchar3(d_frame_in[(y0*w_in+x0)*3], 
                                          d_frame_in[(y0*w_in+x0)*3+1],
                                          d_frame_in[(y0*w_in+x0)*3+2]);
                uchar3 x1y0 = make_uchar3(d_frame_in[(y0*w_in+x1)*3], 
                                          d_frame_in[(y0*w_in+x1)*3+1],
                                          d_frame_in[(y0*w_in+x1)*3+2]);
                uchar3 x0y1 = make_uchar3(d_frame_in[(y1*w_in+x0)*3], 
                                          d_frame_in[(y1*w_in+x0)*3+1],
                                          d_frame_in[(y1*w_in+x0)*3+2]);
                uchar3 x1y1 = make_uchar3(d_frame_in[(y1*w_in+x1)*3], 
                                          d_frame_in[(y1*w_in+x1)*3+1],
                                          d_frame_in[(y1*w_in+x1)*3+2]);

                //calculate interpolated value as normalized directly
                float tj = t*j;
                float sj = s*j;
                float ti = t*i;
                float si = s*i;
                float r = (x0y0.z*tj+x0y1.z*sj+x1y0.z*ti+x1y1.z*si)/255.0f;
                float g = (x0y0.y*tj+x0y1.y*sj+x1y0.y*ti+x1y1.y*si)/255.0f;
                float b = (x0y0.x*tj+x0y1.x*sj+x1y0.x*ti+x1y1.x*si)/255.0f;
                
                //write as RGB CHW directly
                //RRR...GGG...BBB
                d_frame_out[pixelbyte_r]=r;
                d_frame_out[pixelbyte_g]=g;
                d_frame_out[pixelbyte_b]=b;
            }

        }
    }
}

void frame_preprocess(uint8_t* d_frame_in,float* d_frame_out,
                        int h_in, int w_in, int h_out, int w_out){
    //wrapper function, given a frame in BGR HWC mode on device, resize frame to given size and 
    //convert to RGB CHW with normalized RGB value
    float scale = min((float)h_out/(float)h_in,(float)w_out/(float)w_in);
    int re_height = floor(h_in*scale);
    int re_width = floor(w_in*scale);
    int padx = (w_out-re_width)/2;
    int pady = (h_out-re_height)/2;
	dim3 dimBlock(16, 16, 1 );
	dim3 dimGrid(ceil(w_out/16), ceil(h_out/16), 1 );
    preprocess<<<dimGrid,dimBlock>>>(d_frame_in,d_frame_out,h_in,w_in,
        h_out,w_out,padx,pady,scale);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("preprocess failed: %s\n", cudaGetErrorString(errora));
    }
}

__device__ float detectionIOU(void){

}

__global__ void Reduce_bestclass(float* d_raw_detection, int Num_raw_detection, int height_raw_detection,
                                float* d_filtered_detect, int* class_id, float threshold){
    //kernel to find class id with best score
    //each thread handle 1 detection and 80 classes

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocksize = blockDim.x*blockDim.y;
    int totalthreads = blocksize*gridDim.x;

    for(int i = idx;i<Num_raw_detection;i+=totalthreads){
        if (idx >= Num_raw_detection) return;
        int   best_class_id   = -1;
        float best_score = 0.0f;

        for (int c = 4; c < height_raw_detection; ++c) {
            float score = d_raw_detection[c * Num_raw_detection + idx];
            if (score > best_score) {
                best_score = score;
                best_class_id   = c - 4;   //0-79
            }
        }
        //write best score at the 1st detection row
        d_filtered_detect[idx]=d_raw_detection[idx];
        d_filtered_detect[Num_raw_detection+idx]=d_raw_detection[Num_raw_detection+idx];
        d_filtered_detect[2*Num_raw_detection+idx]=d_raw_detection[2*Num_raw_detection+idx];
        d_filtered_detect[3*Num_raw_detection+idx]=d_raw_detection[3*Num_raw_detection+idx];
        if (best_score>threshold){
            d_filtered_detect[4*Num_raw_detection+idx]=best_score;
            class_id[idx]=best_class_id;
        }else{
            d_filtered_detect[4*Num_raw_detection+idx]=0.0;
            class_id[idx]=-1;
        }
    }
}


__global__ void Sort_by_confidence(float* d_filtered_detect, int Num_raw_detection,
                                int* class_id, int class_count){
    int tid = blockDim.x*threadIdx.y+threadIdx.x;
    int blocksize = blockDim.x*blockDim.y;
	int warpId = tid >> 5;
    int lane   = tid & 31;
    //each block 1 class, eahc thread 1 detection col
    //num of detection/32 = shared mnem needed
    __shared__ float max[384];
    __shared__ int max_source[384];
    for(int c_id = blockIdx.x;c_id<class_count;c_id+=gridDim.x){
        for(int i = tid;i<Num_raw_detection;i+=blocksize){
            if(tid<384){
                max[tid]=-INFINITY;
            }
            warpId+=i>>5;
            int source_tid = tid;
            if(class_id[i]==c_id){
                float score = d_filtered_detect[4*Num_raw_detection+i];
            }else{
                float score = -INFINITY;
            }
            //find a maximum for each wrap and record the source
            for (int offset = 16; offset > 0; offset /= 2) {
                other_score = __shfl_down_sync(0xffffffff, score, offset);
                other_id = __shfl_down_sync(0xffffffff, source_tid, offset);
                if(other_score>score){
                    socre = other_score;
                    source_tid = other_id;
                }
            }
            //update to shared mem
            if(lane==0){max[warpId]=score;max_source[warpId]=source_tid;}
        }
        __syncthreads();
        for(int i = tid;i<Num_raw_detection;i+=blocksize){
            
        }

    }
}

int NMS(float* d_raw_detections, int* detection_shape, float* d_final_detections, int* class_id){
    //wrapper function, given the output from models, filter and extract the detections for SORT
    //model output tensor 1*84*8400, row major
    //output to [5*M], each col, 4 coordinates value + 1 class id
    int Num_raw_detection = detection_shape[2];  //number of detections 8400
    int height_raw_detection = detection_shape[1];  //number of 0:3 coordinates, 4-83, score of each class

    dim3 block(256,1,1);
    dim3 grid(ceil((Num_raw_detection+255)/256),1,1);
    Reduce_bestclass<<<grid,block,sizeof(int)*(height_raw_detection-4)>>>
    (d_raw_detections,Num_raw_detection,height_raw_detection,d_final_detections,class_id,0.25);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("detection extract failed: %s\n", cudaGetErrorString(errora));
    }
}