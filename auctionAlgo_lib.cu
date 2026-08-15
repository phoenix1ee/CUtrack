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


__global__ void updateValueAndBid(float* d_IOU, float* d_bid, float* d_price, int* d_bid_target,
								int* d_track_assignment, 
								int activeTracks, int detections){
	//calculate the value and calculate bid(write to price)
	//each block handle 1 track
	//each thread handle 1 detection
	float ep = 1.0f/activeTracks;
	//support up to 8192 blocksize / detections ~ yolo max, and unlikely
	__shared__ float sharedmax[256];
	__shared__ float sharedmax2[256];
	
	__shared__ float sharedID[256];
	
	int blocksize = blockDim.x*blockDim.y;
    int totalblocks = gridDim.x;
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
	int warpId = tid >> 5;
    int lane   = tid & 31;
	int numWarps = (blocksize + 31) / 32;

	//if(tid>detections){return;}
	if(blockIdx.x>=activeTracks)return;
	for(int i = blockIdx.x;i<activeTracks;i+=totalblocks){
		if (d_track_assignment[i] != -1) continue; // tracks with already assigned detections, don't bid!
		if(tid<256){
			sharedmax[tid]=0.0;
			sharedmax2[tid]=0.0;
			sharedID[tid]=-1;
		}
		float local_max = 0.0;
		float local_2ndmax = 0.0;
		int max_d_id = -1;
		float benefit = 0;
		float price = 0;
		float v=0;
		for(int j = tid;j<detections;j+=blocksize){
			//calculate profit vij = IOUij-pj
			benefit = d_IOU[i*detections+j];
			//if (benefit<0.3){benefit=-1e3;}
			price = d_price[j];
			v=benefit-price;
			if(v>local_max){
				local_2ndmax = local_max;
				local_max = v;
				max_d_id = j;
			}else if(v>local_2ndmax){
				local_2ndmax = v;
			}
		}
		//find a max and 2nd max for each wrap
		for (int offset = 16; offset > 0; offset /= 2) {
			float remote_max = __shfl_down_sync(0xffffffff, local_max, offset);
			int remote_max_id = __shfl_down_sync(0xffffffff, max_d_id, offset);

			float remote_max2 = __shfl_down_sync(0xffffffff, local_2ndmax, offset);
			if(remote_max>local_max){
				local_2ndmax = fmaxf(local_max, remote_max2);
				local_max = remote_max;
				max_d_id = remote_max_id;
			}else{
				local_2ndmax = fmaxf(local_2ndmax,remote_max);
			}
		}
		//update to shared mem
		if(lane==0){
			sharedmax[warpId]=local_max;
			sharedID[warpId]=max_d_id;
			sharedmax2[warpId]=local_2ndmax;
			
		}
		__syncthreads();
		//read from shared mem and find block level max and 2nd max
		if(warpId==0){
			local_max = (lane < numWarps) ? sharedmax[lane] : 0.0;
			max_d_id = (lane < numWarps) ? sharedID[lane] : -1;
			local_2ndmax = (lane < numWarps) ? sharedmax2[lane] : 0.0;
			
			for (int offset = 16; offset > 0; offset = offset /2) {
				float remote_max = __shfl_down_sync(0xffffffff, local_max, offset);
				int remote_max_id = __shfl_down_sync(0xffffffff, max_d_id, offset);

				float remote_max2 = __shfl_down_sync(0xffffffff, local_2ndmax, offset);
				if(remote_max>local_max){
					local_2ndmax = fmaxf(local_max, remote_max2);
					local_max = remote_max;
					max_d_id = remote_max_id;
				}else{
					local_2ndmax = fmaxf(local_2ndmax,remote_max);
				}
			}
			if (lane == 0) {
				sharedmax[0]=local_max;
				sharedID[0]=max_d_id;
				sharedmax2[0]=local_2ndmax;
            }
		}
		__syncthreads();
		max_d_id = sharedID[0];
		if (max_d_id != -1) {
			float bid = d_price[max_d_id]+(sharedmax[0]-sharedmax2[0])+ep;
			d_bid[i]=bid;
			d_bid_target[i]=max_d_id;
		}
	}
}

__global__ void resolveAssignment(const float* d_bids, const int* d_bid_target, float* d_price, 
					int* d_track_assignments,     int* d_det_assignments,      
    				const int activeTracks, const int num_dets,
    				bool* d_any_changes) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_dets) return;

	float max_bid = -1.0f;
	int winning_track = -1;
	// One thread per Detection j
	for (int i = 0; i < activeTracks; i++) {
		if (d_bid_target[i] == j) { // Does this track want me?
			float current_bid = d_bids[i];
			if (current_bid > max_bid) {
				max_bid = current_bid;
				winning_track = i;
			}
		}
	}
    // If there was a valid bid, resolve the competitive bid of other tracks
    if (winning_track != -1 && (max_bid - (float)d_price[j])>1e-6) {
        
        // Unassign the OLD track that used to own this detection
        int old_track = d_det_assignments[j];
        if (old_track != -1) {
            d_track_assignments[old_track] = -1; 
        }

        // Assign the NEW winning track
        d_price[j] = max_bid; // Assuming price is stored as int/fixed-point
        d_det_assignments[j] = winning_track;
        d_track_assignments[winning_track] = j;

        *d_any_changes = true;
    }
}

__global__ void initialPriceAndAssignment(float* d_price, int*d_match_track, int activetracks,
				int*d_match_detection, int detections){
	//reach thread handle 1 detection
	//initialize price of each detection to 0.0 and match to -1
	int blocksize = blockDim.x*blockDim.y;
    int totalthreads = gridDim.x*blocksize;
    int tid = blockIdx.x*blocksize+threadIdx.y*blockDim.x+threadIdx.x;
    for(int i = tid;i<detections;i+=totalthreads){
		if(i>=detections)continue;
    	d_price[i] = 0.0;
		d_match_detection[i] = -1;
	}
    for(int i = tid;i<activetracks;i+=totalthreads){
		d_match_track[i] = -1;
	}
}


void auction_assignment(tracker &tracker,int width_detections, int height_tracks){
	//Wrapper function to perform assignment of detection to tracks
	//The Auction Algorithm for assignment
	float* d_IOU = tracker.d_IOU;
	float* d_price = tracker.d_price;
	int* d_match_track = tracker.d_match_track;
	int* d_match_detection = tracker.d_match_detections;
	float* d_bid = tracker.d_bid;
	int* d_bid_target = tracker.d_bid_target;
	//define grid and block size
	dim3 dimBlock(32, 8, 1 );
	dim3 dimGrid((width_detections*height_tracks+255)/256, 1, 1 );
	//tracks Assignment
	//build price matrix and detection match
	initialPriceAndAssignment<<<(width_detections+255)/256,256>>>(d_price,d_match_track,height_tracks,d_match_detection,width_detections);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error !=cudaSuccess){
		printf("price matrix and detection match initialization kernel failed: %s", cudaGetErrorString(error));
	}
	int itr = 0;
	int max_itr = 50;
	bool h_any_changes=true;
	bool* d_any_changes;
	cudaMalloc((void**)&d_any_changes,sizeof(bool));
	while(h_any_changes && itr<max_itr){
		cudaMemset(d_bid, 0, sizeof(float)*tracker.Max_detection*tracker.Max_Tracks);
		cudaMemset(d_bid_target, -1, sizeof(int)*tracker.Max_Tracks);
		
		//calculate bid
		updateValueAndBid<<<height_tracks,256>>>(d_IOU,d_bid,d_price,d_bid_target,d_match_track,height_tracks,width_detections);
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if (error !=cudaSuccess){
			printf("updateValueAndBid kernel failed: %s", cudaGetErrorString(error));
			break;
		}
		//matching
		cudaMemset(d_any_changes, 0, sizeof(bool));
		resolveAssignment<<<(width_detections+255)/256,256>>>(d_bid,d_bid_target,d_price,d_match_track,d_match_detection,height_tracks,width_detections,d_any_changes);
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if (error !=cudaSuccess){
			printf("updateValueAndBid kernel failed: %s", cudaGetErrorString(error));
			break;
		}
		//sync the status
		cudaMemcpy(&h_any_changes, d_any_changes, sizeof(bool), cudaMemcpyDeviceToHost);
		itr++;
	}
	cudaFree(d_any_changes);
}