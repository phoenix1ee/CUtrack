#include "include/sort_lib.h"
#include "include/helper.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>

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

__device__ float boxIOU(float acx,float acy,float aw,float ah,
                        float bcx,float bcy,float bw,float bh){
    //a/b->[cx,cy,w,h]
    
    float ax1=acx-aw/2;
    float ax2=acx+aw/2;
    float ay1=acy-ah/2;
    float ay2=acy+ah/2;

    float bx1=bcx-bw/2;
    float bx2=bcx+bw/2;
    float by1=bcy-bh/2;
    float by2=bcy+bh/2;

    float xx1 = fmaxf(ax1, bx1);
    float yy1 = fmaxf(ay1, by1);
    float xx2 = fminf(ax2, bx2);
    float yy2 = fminf(ay2, by2);

    float w = max(0.0f, xx2 - xx1);
    float h = max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = aw*ah;
    float area_b = bw*bh;

    return inter / (area_a + area_b - inter + 1e-6f);
}

__global__ void NMS_refine(float* d_final_detection,float* d_buffer_detection, 
                            int* d_final_class_id,
                           int num_raw_detection,int num_class, float IOU_threshold, int* d_valid_count){
    //find overlapped detection and suppresss, input is sorted with decending order in score
    //each thread handle 1 detection
    int blockid = blockIdx.y*gridDim.x+blockIdx.x;
    int blocksize = blockDim.x*blockDim.y;
    int totalthreads = gridDim.x*gridDim.y*blocksize;
    int tid = blockid*blocksize+threadIdx.y*blockDim.x+threadIdx.x;
    if (tid>num_raw_detection){return;}
    int count=0;
    for(int i = 0;i<num_raw_detection;i++){  //loop through all detections as subject
        int subj_class = d_final_class_id[i];
        if (subj_class==-1 && tid ==0){
            d_buffer_detection[num_raw_detection*4+i]=-1;
            continue;
        } //only execute if subject class != -1
        //update the subject detection score if class is valid
        if(tid==0){
            d_buffer_detection[num_raw_detection*4+i]=d_final_detection[num_raw_detection*4+i];
            count++;
        }
        for(int j = tid+i+1;j<num_raw_detection;j+=totalthreads){
            int current_class = d_final_class_id[j];
            if(current_class==-1){
                d_buffer_detection[num_raw_detection*4+j]=-1.0;
            }else if(current_class!=subj_class){
                continue;
            }else{
                //only execute if compare class = subj_class
                float acx = d_final_detection[i];
                float acy = d_final_detection[num_raw_detection+i];
                float aw = d_final_detection[num_raw_detection*2+i];
                float ah = d_final_detection[num_raw_detection*3+i];
                float bcx = d_final_detection[j];
                float bcy = d_final_detection[num_raw_detection+j];
                float bw = d_final_detection[num_raw_detection*2+j];
                float bh = d_final_detection[num_raw_detection*3+j];

                float IOU = boxIOU(acx,acy,aw,ah,bcx,bcy,bw,bh);
                if (IOU>IOU_threshold){
                    //use the buffer as staging for result
                    //suppress by setting score = -1 and class = -1
                    d_buffer_detection[num_raw_detection*4+j]=-1.0;
                    d_final_detection[num_raw_detection*4+j]=-1.0;
                    d_final_class_id[j]=-1;
                }else{
                    d_buffer_detection[num_raw_detection*4+j]=d_final_detection[num_raw_detection*4+j];
                }
            }
        }
        __syncthreads();
    }
}


__global__ void Reduce_bestclass(float* d_raw_detection, float* d_filter_detection,
                                int Num_raw_detection, int height_raw_detection,
                                 int* d_raw_class_id, float threshold){
    //kernel to find class id with best score
    //each thread handle 1 detection and 80 classes
    //d_raw_detection is 84*8400, d_filtered_detect is 5*8400, class_id is 1*8400
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
        //write best score to raw detection row 4
        if (best_score>threshold){
            d_filter_detection[4*Num_raw_detection+idx]=best_score;
            d_raw_class_id[idx]=best_class_id;
        }else{
            d_filter_detection[4*Num_raw_detection+idx]=-1.0;
            d_raw_class_id[idx]=-1;
        }
    }
}

__global__ void update_after_sort(float* d_detection_in, float* d_detection_out, int* d_class_in, int* d_class_out,
                                    int* order, int width, int num_rows) {
    //copy the coordinates to output
    // global ID
    int global_tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) 
                     + (threadIdx.y * blockDim.x + threadIdx.x);
    int grid_stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    // Loop over columns
    for (int col = global_tid; col < width; col += grid_stride) {
        // Use the sorted index found by Thrust
        int source_col = order[col]; 
        // Loop every row for a column
        for (int row = 0; row < num_rows; row++) {
            int in_idx  = (row * width) + source_col;
            int out_idx = (row * width) + col;
            //update
            d_detection_out[out_idx] = d_detection_in[in_idx];
        }
        //update class id to final
        d_class_out[col]=d_class_in[source_col];
    }
}


void NMS(float* d_raw_detections, int* d_raw_class_id, int Num_raw_detection, int height_raw_detection, 
        float* d_buffer_detections, int* d_buffer_class_id, int*d_detection_count){
    //wrapper function, given the output from models, filter and extract the detections for SORT
    //model output tensor 1*84*8400, row major
    //output to the same raw detection array at row 0-4, each col, 4 coordinates value + 1 score

//find best class
    dim3 block(256,1,1);
    dim3 grid((Num_raw_detection+255)/256,1,1);
    Reduce_bestclass<<<grid,block>>>(d_raw_detections,d_buffer_detections,Num_raw_detection,height_raw_detection,d_raw_class_id,0.25);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("best class kernel failed: %s\n", cudaGetErrorString(errora));
    }

//sort using thrust
    // Initialize indices: [0, 1, 2, 3]
    thrust::device_vector<int> col_indices(Num_raw_detection);
    thrust::sequence(col_indices.begin(), col_indices.end());
    // Get a pointer to the start of the specific row acting as the key
    thrust::device_ptr<float> thrust_d_buffer_detection(d_buffer_detections);
    auto key_row_begin = thrust_d_buffer_detection + (4 * Num_raw_detection);
    thrust::sort_by_key(key_row_begin, key_row_begin + Num_raw_detection, col_indices.begin(),thrust::greater<float>());
    cudaDeviceSynchronize();
    //get raw pointer of results
    int* order = thrust::raw_pointer_cast(col_indices.data());
    //copy the coordinates and class to final detection under descending order of score
    update_after_sort<<<grid,block>>>(d_raw_detections,d_buffer_detections,
                                      d_raw_class_id,d_buffer_class_id,order,
                                      Num_raw_detection,4);
    cudaDeviceSynchronize();
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("final detection update kernel failed: %s\n", cudaGetErrorString(errora));
    }
//suppress duplicate with detection IOU
    //use the raw array as buffer
    NMS_refine<<<grid,block>>>(d_buffer_detections,d_raw_detections,
                                d_buffer_class_id,
                                Num_raw_detection,height_raw_detection-4,0.4,d_detection_count);
    cudaDeviceSynchronize();
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("suppression of final detection kernel failed: %s\n", cudaGetErrorString(errora));
    }

//re-sort and count number of output
    // re-Initialize indices of last thrust
    thrust::sequence(col_indices.begin(), col_indices.end());
    // Get a pointer to the start of the specific row acting as the key
    thrust::device_ptr<float> thrust_d_suppressed_detection(d_raw_detections);
    key_row_begin = thrust_d_suppressed_detection + (4 * Num_raw_detection);
    thrust::sort_by_key(key_row_begin, key_row_begin + Num_raw_detection, col_indices.begin(),thrust::greater<float>());
    cudaDeviceSynchronize();
    //copy the coordinates and class back to raw detection under descending order of score
    update_after_sort<<<grid,block>>>(d_buffer_detections,d_raw_detections,
                                      d_buffer_class_id,d_raw_class_id,order,
                                      Num_raw_detection,4);
    cudaDeviceSynchronize();
    errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("final detection update kernel failed: %s\n", cudaGetErrorString(errora));
    }
}