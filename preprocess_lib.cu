#include "include/sort_lib.h"
#include "include/helper.h"

__global__ void HWCtranspose(uint8_t*a,uint8_t*b,int matrixwidth, int matrixheight){
	//do transpose

	//a tile of 32*32
	__shared__ uint8_t sharedArray[32][33];
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


__global__ void BGRconversionsimple(uint8_t* d_frame_in,int totalpixel){
    //blocksize * 3 bytes of shared mem
    //Convert BGR → RGB
    int blocksize = blockDim.x*blockDim.y;
    int gridsize = gridDim.x*gridDim.y;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int blockID=blockIdx.y*gridDim.x+blockIdx.x;
    int offset=blockID*blocksize;
    if (offset>totalpixel){return;}
    //each thread handle 1 pixel
    for(int i=offset;i<totalpixel;i+=gridsize*blocksize){
        if(i+tid<totalpixel){
            
            uint8_t B=d_frame_in[i*3+tid*3];
            uint8_t G=d_frame_in[i*3+tid*3+1];
            uint8_t R=d_frame_in[i*3+tid*3+2];

            d_frame_in[i*3+tid*3]=R;
            d_frame_in[i*3+tid*3+1]=G;
            d_frame_in[i*3+tid*3+2]=B;
        }
    }
}

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

            int pixelbyte_r= pixel_y*out_w+pixel_x;
            int pixelbyte_b= out_w*out_h+pixel_y*out_w+pixel_x;
            int pixelbyte_g= out_w*out_h*2+pixel_y*out_w+pixel_x;

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
    //wrapper function, given a frame in BGR HWC mode, resize frame to given size and 
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

void frame_BGRtoRGB(uint8_t* d_frame_in,int totalpixel){
    //wrapper function, convert a frame in BGR HWC to RGB HWC
	dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((totalpixel+255)/256, 1, 1 );
    BGRconversionsimple<<<dimGrid,dimBlock>>>(d_frame_in,totalpixel);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("BGR to RGB failed: %s\n", cudaGetErrorString(errora));
    }
}

void frame_HWCtoCHW(uint8_t*d_input,uint8_t*d_output,int width, int height){
    //wrapper function, convert a frame from HWC to CHW
    dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((width*height+255)/256, 1, 1 );
    HWCtranspose<<<dimGrid,dimBlock>>>(d_input,d_output,width,height);
    cudaDeviceSynchronize();
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("HWC to CHW failed: %s\n", cudaGetErrorString(errora));
    }
}