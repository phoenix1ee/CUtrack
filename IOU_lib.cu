#include "include/helper.h"
#include "include/sort_lib.h"

__device__ float boxIOUxysr(float acx,float acy,float as,float ar,
                        float bcx,float bcy,float bs,float br){
    //a/b->[cx,cy,s,r]
    float aw = sqrt(as*ar);
    float ah = sqrt(as/ar);
    float bw = sqrt(bs*br);
    float bh = sqrt(bs/br);
    
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

    return inter / (as + bs - inter + 1e-6f);
}


__global__ void dummykernel(void){}

__global__ void computeIOUmatrix2(float* d_predictedstate, float* d_detectbox, float* d_IOUmatrix, 
    int Ntracks, int Mdetection,int image_w, int image_h){
    //kernel to compute IOU cost matrix
    //d_predictedstate= [x, y, s, r, x., y., s.] 7 values * N tracks, all floats, column major
    //d_detectbox = [x_center, y_center, w width, h height] all floats, 4 values * M detections, row major
    //assume blockDim.x=32 and blocksize is multiple of 32
    //each block handles a width of 32 
    //each thread handle 1 pair of tracks i detection j
    //let d_IOUmatrix to be written as N*M
    //calculate id
    int threadoffset_j = blockIdx.x*blockDim.x+threadIdx.x;
    int threadoffset_i = blockIdx.y*blockDim.y+threadIdx.y;
	int gridwidth = gridDim.x*blockDim.x;
    int gridheight = gridDim.y*blockDim.y;

    if (threadoffset_i >= Ntracks || threadoffset_j>=Mdetection) return;

    for(int i=threadoffset_i; i<Ntracks; i+=gridheight){
        for(int j=threadoffset_j; j<Mdetection; j+=gridwidth){
            //convert predicted state to predict box (x1, y1, x2, y2)
            //pixel coordinates
            //(x1,y1)-------(x2,y1)
            //   |             |
            //   |             |
            //(x1,y2)-------(x2,y2)     
            float s_p = d_predictedstate[i*7+2];
            float w_p=sqrt(s_p*d_predictedstate[i*7+3]);
            float h_p=s_p/w_p;

            float x_p = d_predictedstate[i*7];
            float y_p = d_predictedstate[i*7+1];

            float x1p=x_p-w_p/2;
            float x2p=x_p+w_p/2;
            float y1p=y_p-h_p/2;
            float y2p=y_p+h_p/2;
            
            //convert detection box to  (x1, y1, x2, y2)  
            float x_d = d_detectbox[j*4]*image_w;
            float y_d = d_detectbox[j*4+1]*image_h;
            float w_d = d_detectbox[j*4+2];
            float h_d = d_detectbox[j*4+3];
            
            float x1d=x_d-w_d/2;
            float x2d=x_d+w_d/2;
            float y1d=y_d-h_d/2;
            float y2d=y_d+h_d/2;

            //calculate IOU value
            float IOU = 0;
            if (abs(x_p-x_d)<(w_d+w_p)/2 && abs(y_p-y_d)<(h_d+h_p)/2){
                //some overlap, pixel 0,0 at top left corner
                float overlap_w = 0;
                if (x1d<x1p && x2p<x2d){
                    //predict box width inside detection
                    overlap_w = w_p;
                }
                else if (x1p<x1d && x2d<x2p){
                    //detect box width inside predict
                    overlap_w = w_d;
                }
                else{
                    overlap_w = min(abs(x1d-x2p),abs(x1p-x2d));
                }
                float overlap_h = 0;
                if (y1d<y1p && y2p<y2d){
                    //predict box height inside detection
                    overlap_h = h_p;
                }
                else if (y1p<y1d && y2d<y2p){
                    //detect box height inside predict
                    overlap_h = h_d;
                }
                else{
                    //detect box and predict box intersect but neither height or width are inside each other
                    overlap_h = min(abs(y1d-y2p),abs(y1p-y2d));
                }
                float overlap_area = overlap_w*overlap_h;
                float total_area = s_p+w_d*h_d-overlap_area;
                IOU = overlap_area / total_area;
                //printf("IOU: %.2f \n",IOU);
            }
            //write to d_IOUmatrix, use row major order at here to fit reduction kernels
            //printf("row: %d col: %d \n",i,j);
            d_IOUmatrix[i*Mdetection+j]=1-IOU;
        }
    }
}

__global__ void computeIOUmatrix(float* d_predictedstate, float* d_detectbox, float* d_IOUmatrix, 
    int Ntracks, int Mdetection, int N, int M, int n, int m){
    //kernel to compute IOU cost matrix
    //d_predictedstate= [x, y, s, r, x., y., s.] n=7 values * N tracks, all floats, row major
    //d_detectbox = [x_center, y_center, s , r] all floats, m=4 values * M detections, row major
    //each thread handle 1 pair of tracks i detection j
    //each block handle 1 track i and all detection pair (i,j)
    //let d_IOUmatrix to be written as N*M, row major
    //calculate id

    int blockid = blockIdx.x;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int blocksize = blockDim.x*blockDim.y;
    int num_block = gridDim.x;
    if (blockid >= Ntracks || tid>=Mdetection) return;

    for(int i=blockid; i<Ntracks; i+=num_block){
        for(int j=tid; j<Mdetection; j+=blocksize){
            //combination: track i and detection j
            float p_x = d_predictedstate[i];
            float p_y = d_predictedstate[N+i];
            float p_s = d_predictedstate[2*N+i];
            float p_r = d_predictedstate[3*N+i];

            float d_x = d_detectbox[j];
            float d_y = d_detectbox[M+j];
            float d_s = d_detectbox[2*M+j];
            float d_r = d_detectbox[3*M+j];

            //calculate IOU value
            float IOU = boxIOUxysr(p_x,p_y,p_s,p_r,d_x,d_y,d_s,d_r);
            //write to d_IOUmatrix, use row major order at here to fit reduction kernels
            d_IOUmatrix[i*Mdetection+j]=IOU;
        }
    }
}

void tracker_compute_IOU2(tracker* tracker, float* d_detectbox, int activetrack, int activedetection, int image_w, int image_h){
    //wrapper function to compute IOU
    
    dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((activedetection*activetrack+255)/256, 1, 1 );
    computeIOUmatrix2<<<dimGrid,dimBlock>>>(tracker->d_state_predicted,d_detectbox,tracker->d_IOU,
        activetrack,activedetection,image_w,image_h);
    cudaDeviceSynchronize();
}

void tracker_compute_IOU(tracker &tracker, int activetrack, int activedetection){
    //wrapper function to compute IOU
    dim3 dimBlock(16, 16, 1 );
	dim3 dimGrid((activetrack+255)/256, 1, 1 );
    computeIOUmatrix<<<dimGrid,dimBlock>>>(tracker.d_state_predicted,tracker.d_Z,tracker.d_IOU,
        activetrack,activedetection,tracker.Max_Tracks,tracker.Max_detection,tracker.n,tracker.m);
    cudaError_t errora = cudaGetLastError();
    if (errora != cudaSuccess){
        printf("IOU compute kernel failed: %s\n", cudaGetErrorString(errora));
    }
	cudaDeviceSynchronize();

}
