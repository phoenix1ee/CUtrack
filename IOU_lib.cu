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
