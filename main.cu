#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "include/helper.h"
#include "include/sort_lib.h"
#include "include/inference.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


int main(int argc, char* argv[]) {
try {
    if (argc != 2) {std::cout<<"wrong argument\n";return 1;}
    //set a max expected detection 2000
    int Max_Tracks=8400;
    int Max_detection = 8400;
    //set state variable and detection spec
    int m=5;     //detection
    int n=7;     //state variable

    //setup a tracker
    tracker tracker1(Max_Tracks,Max_detection,m,n);
    tracker1.allocateOnDevice();
    int detection_count[1] = {0};
    int trackcount = 0;
    int displaycount = 0;
    int* d_detection_count=tracker1.d_totaldetections;
    int* d_track_count=tracker1.d_totaltracks;
    int* d_displaycount = tracker1.d_goodtracks;
    set_single_F(tracker1);
    set_single_Q(tracker1);
    set_single_R(tracker1);
    set_single_H(tracker1);
    //----------------------------------------------------------------------
    //setup inference model
    YoloDetector detector;
    detector.load_model(L"yolo11n.onnx");
    detector.extract_model_data();

    //detector output memory
    float* d_detector_output;
    size_t detector_output_size = detector.output_shapes_size[0]*sizeof(float);
    cudaMalloc((void**)&d_detector_output,detector_output_size);
    //----------------------------------------------------------------------
    //handle input video stream
    cv::VideoCapture cap(argv[1]);
    //std::cout<<"finish video capture"<<std::endl;
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));



    //std::cout<<"video width:"<<width<<" height: "<<height<<std::endl;
    // memory for the captured frame
    cv::Mat temp_frame;
    cap.read(temp_frame);  //read 1 frame
    if (temp_frame.empty()) {
        std::cout << "Failed to read video stream!\n";
        return -1;
    }
    size_t frame_size = temp_frame.total() * temp_frame.elemSize();

    //Manually allocate Pinned (Page-Locked) Host Memory
    uint8_t* h_pinned_ptr;
    cudaError_t err = cudaHostAlloc((void**)&h_pinned_ptr, frame_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        printf("Failed to allocate pinned memory: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wrap a cv::Mat header around your pinned memory pointer
    cv::Mat pinned_frame(height, width, CV_8UC3, h_pinned_ptr);

    //allocate device mem for the frame
    uint8_t* d_frame_ptr;
    cudaMalloc(&d_frame_ptr, frame_size);
    std::cout<<"created device frame buffer"<<std::endl;
    
    // setup output memory for the preprocessing
    float* d_preprocess_output;
    int out_width = detector.input_shapes[0][3];
    int out_height = detector.input_shapes[0][2];
    size_t data_size_after_preprocess = sizeof(float)*out_height*out_width*3;
    cudaMalloc((void**)&d_preprocess_output,data_size_after_preprocess);
    //----------------------------------------------------------------------
    //create the tensor for the model
    detector.create_input_tensor(d_preprocess_output,data_size_after_preprocess);
    detector.create_output_tensor(d_detector_output,detector_output_size);
    detector.bind_tensors();
    std::cout<<"created detector tensors"<<std::endl;
    //set up memory for post processing
    size_t Num_raw_detection = detector.output_shapes[0][2];//number of detections 8400
    int height_raw_detection = detector.output_shapes[0][1];//number of 0:3 coordinates
    float* d_detection_buffer = tracker1.d_Z;
    int* d_class_id;
    cudaMalloc((void**)&d_class_id,Num_raw_detection*sizeof(int)*2);
    int* d_class_id_buffer = d_class_id+Num_raw_detection;
    //----------------------------------------------------------------------
    // host Buffer for GPU to copy detected boxes back
    float* h_box_buffer = (float*)calloc(Max_Tracks*4,sizeof(float));
    err = cudaHostRegister(h_box_buffer, Max_Tracks*4*sizeof(float), cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        printf("host output box buffer register failed: %s", cudaGetErrorString(err));
    }
    float* d_output_buffer = tracker1.d_state_output;

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    bool first_frame = true;
    int num_frame = 1;
    while (true) {
        //copy the frame extracted to pinned memory
        temp_frame.copyTo(pinned_frame);
        // A. Transfer frame to GPU
        cudaMemcpy(d_frame_ptr, h_pinned_ptr, frame_size, cudaMemcpyHostToDevice);
        // B. Run processing kernel 
        // frame preprocessing
        frame_preprocess(d_frame_ptr,d_preprocess_output,height,width,out_height,out_width);
        // then start inference
        detector.run();
        // post-processing
        NMS(d_detector_output,d_class_id,Num_raw_detection,height_raw_detection,d_detection_buffer,d_class_id_buffer,d_detection_count);
        //update host detection counts
        cudaMemcpy(detection_count,d_detection_count,sizeof(int),cudaMemcpyDeviceToHost);
        // copy/convert the detections to tracker's measurement matrix buffer
        copyToTracker(d_detector_output,tracker1.d_Z,Num_raw_detection,detection_count[0]);
        //calculate states
        printf("frame %d from total: %d display: %d detect: %d\n",num_frame,trackcount,displaycount, detection_count[0]);
        if (first_frame){
            // add the new initial state
            set_first_state(tracker1,0,detection_count[0]);
            set_first_Pcov(tracker1,0,detection_count[0]);
            set_first_age_hit_status(tracker1,0,detection_count[0]);
            cudaMemcpy(d_track_count,d_detection_count,sizeof(int),cudaMemcpyDeviceToDevice);
            trackcount = detection_count[0];
            first_frame = !first_frame;
        }else{  //for each frame after 1st frame
            //prediction
            make_state_prediction(tracker1,trackcount);
            make_cov_prediction(tracker1,trackcount);
            //compute IOU and cost matrixbetween predicted box and detection
            tracker_compute_IOU(tracker1,trackcount,detection_count[0]);
            //hungarian assignment
            auction_assignment(tracker1,detection_count[0],trackcount);
            //Calculate Kalman gain for all tracks
            tracker_kalman_gain(&tracker1,trackcount);
            //update the tracks states
            update_states_Kalman(tracker1,trackcount);
            update_Pcov(tracker1, trackcount);
            //update tracks status, hit streak and age of tracks
            //matched->increase hit, Unmatched track→ track age++
            update_track_status(tracker1,trackcount);            
            //update device total active track count for status 1 and 2 tracks
            update_track_count(tracker1);

            //prepare output confirmed tracks
            update_output_buffer(tracker1,trackcount,out_width,out_height,width,height);
            //cleanup
            rearrangetracks(tracker1, trackcount,num_frame);

            //Create new tracks for unmatched detections
            //use new track counts after delete on device, and added new track to the back, update trackcount
            add_new_tracks(tracker1,detection_count[0]);
            cudaMemcpy(&trackcount,d_track_count,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(&displaycount,d_displaycount,sizeof(int),cudaMemcpyDeviceToHost);
        }
        printf("become total: %d display: %d \n",trackcount,displaycount);
        // C. Transfer ONLY the results of detection box back
        // need to modify the output from tracker before transfer because states are [cx,cy,s,r]
        cudaMemcpy(h_box_buffer, d_output_buffer, sizeof(float)*4*Max_Tracks, cudaMemcpyDeviceToHost);

        // D. Synchronize stream before drawing to ensure boxes are ready
        //cudaStreamSynchronize(stream);

        // E. Visualization (on the Host pinned frame)
        for (int i = 0; i < displaycount; ++i) {
            // Calculate indices based on row-major 4xN structure
            // Index = (row_index * number_of_columns) + column_index
            float x1 = h_box_buffer[0 * Max_Tracks + i];
            float y1 = h_box_buffer[1 * Max_Tracks + i];
            float x2 = h_box_buffer[2 * Max_Tracks + i];
            float y2 = h_box_buffer[3 * Max_Tracks + i];

            // Skip logic: check for empty/invalid boxes
            if (x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0) continue;

            // 2. Draw the rectangle
            cv::rectangle(
                temp_frame, 
                cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 
                cv::Point(static_cast<int>(x2), static_cast<int>(y2)), 
                cv::Scalar(0, 255, 0), 2
            );
        }

        cv::imshow("Demonstration", temp_frame);
        if (cv::waitKey(1) == 27) break; // Exit on ESC
        cap.read(temp_frame);
        if (temp_frame.empty()) break;
        num_frame++;
    }

    // Cleanup...
    cudaFreeHost(h_pinned_ptr);
    cudaHostUnregister(h_box_buffer);
    free(h_box_buffer);
    cudaFree(d_frame_ptr);
    cudaFree(d_preprocess_output);
    cudaFree(d_detector_output);
    cudaFree(d_detection_buffer);
    cudaFree(d_class_id);
    cudaFree(d_detection_count);

    tracker1.freeOnDevice();
    std::cout<<"process done"<<std::endl;
    return 0;
    } catch (const Ort::Exception& e) {
        printf("ONNX Error: %s\n", e.what());
    }
}