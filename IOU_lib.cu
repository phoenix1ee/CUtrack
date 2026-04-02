#include "include/helper.h"
#include "include/sort_lib.h"

__global__ void dummykernel(void){}

__global__ void computeIOUmatrix(float* d_predictedstate, float* d_detectbox, float* d_IOUmatrix, 
    int Ntracks, int Mdetection,int image_w, int image_h){
    //kernel to compute IOU cost matrix
    //d_predictedstate= [x, y, s, r, x., y., s.] 7 values * N tracks, all floats, column major
    //d_detectbox = [x_center, y_center, w width, h height] all floats, 4 values * M detections, column major
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

void tracker_compute_IOU(tracker* tracker, float* d_detectbox, int activetrack, int activedetection, int image_w, int image_h){
    //wrapper function to compute IOU
    dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((activedetection*activetrack+255)/256, 1, 1 );
    computeIOUmatrix<<<dimGrid,dimBlock>>>(tracker->d_state_predicted,d_detectbox,tracker->d_IOU,
        activetrack,activedetection,image_w,image_h);
    cudaDeviceSynchronize();
}