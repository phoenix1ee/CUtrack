#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "include/helper.h"
#include "include/sort_lib.h"
#include "include/inference.h"
#include <onnxruntime_cxx_api.h>

void set_preprocess_mem(ImageData &image,YoloDetector &detector,
                    uint8_t** d_preprocess_input){
}

void free_preprocess_mem(uint8_t* d_preprocess_input,float* d_preprocess_output){
    cudaFree(d_preprocess_input);
    cudaFree(d_preprocess_output);
}

int main(int argc, char** argv){
try {
    if (argc != 3) {std::cout<<"wrong argument\n";return 1;}

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
    int* d_detection_count;
    cudaMalloc((void**)&d_detection_count,sizeof(int));
    set_single_F(tracker1);
    set_single_Q(tracker1);
    set_single_R(tracker1);
    set_single_H(tracker1);

    //setup input stream
    std::string path = argv[1];
    ImageData image = load_jpeg_bgr_hwc_to_host(path);

    std::string path2 = argv[2];
    ImageData image2 = load_jpeg_bgr_hwc_to_host(path2);

    //setup inference model
    YoloDetector detector;
    detector.load_model(L"yolo11n.onnx");
    detector.extract_model_data();

    //set up the memory for preprocessing and detector.
    int width = image.width;
    int height = image.height;
    size_t data_size_to_preprocess = image.size;

    int out_width = detector.input_shapes[0][3];
    int out_height = detector.input_shapes[0][2];
    size_t data_size_after_preprocess = sizeof(float)*out_height*out_width*3;

    uint8_t* d_preprocess_input;
    float* d_preprocess_output;
    cudaMalloc((void**)&d_preprocess_input,data_size_to_preprocess);
    cudaMalloc((void**)&d_preprocess_output,data_size_after_preprocess);

    float* d_detector_output;
    size_t detector_output_size = detector.output_shapes_size[0]*sizeof(float);
    cudaMalloc((void**)&d_detector_output,detector_output_size);

    //create the tensor for the model
    detector.create_input_tensor(d_preprocess_output,data_size_after_preprocess);
    detector.create_output_tensor(d_detector_output,detector_output_size);
    detector.bind_tensors();
    
    //set up memory for post processing
    size_t Num_raw_detection = detector.output_shapes[0][2];//number of detections 8400
    int height_raw_detection = detector.output_shapes[0][1];//number of 0:3 coordinates
    float* d_detection_buffer = tracker1.d_Z;
    int* d_class_id;
    cudaMalloc((void**)&d_class_id,Num_raw_detection*sizeof(int)*2);
    int* d_class_id_buffer = d_class_id+Num_raw_detection;


    bool first_frame = true;
    // should be in a loop
    cudaMemcpy(d_preprocess_input,image.data,data_size_to_preprocess,cudaMemcpyHostToDevice);
    frame_preprocess(d_preprocess_input,d_preprocess_output,height,width,out_height,out_width);
    // then start inference
    detector.run();
    // post-processing
    NMS(d_detector_output,d_class_id,Num_raw_detection,height_raw_detection,d_detection_buffer,d_class_id_buffer,d_detection_count);
    //update host detection counts
    cudaMemcpy(detection_count,d_detection_count,sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"detection count:"<<detection_count[0]<<std::endl;
    // copy/transpose the detection to tracker's measurement matrix
    //transposeArray(d_detector_output,d_detection_buffer,Num_raw_detection,5);
    //
    //cudaMemcpy(tracker1.d_Z,d_detector_output,sizeof(float)*tracker1.m*Num_raw_detection,cudaMemcpyDeviceToDevice);
    copyToTracker(d_detector_output,tracker1.d_Z,Num_raw_detection,detection_count[0]);
    //calculate states
    //if (first_frame){
        // add the new initial state
        set_first_state(tracker1,0,detection_count[0]);
        set_first_Pcov(tracker1,0,detection_count[0]);
        trackcount = detection_count[0];
        first_frame = !first_frame;
        make_prediction(tracker1,trackcount);

    cudaMemcpy(d_preprocess_input,image2.data,data_size_to_preprocess,cudaMemcpyHostToDevice);
    frame_preprocess(d_preprocess_input,d_preprocess_output,height,width,out_height,out_width);
    detector.run();
    NMS(d_detector_output,d_class_id,Num_raw_detection,height_raw_detection,d_detection_buffer,d_class_id_buffer,d_detection_count);
    cudaMemcpy(detection_count,d_detection_count,sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"detection count:"<<detection_count[0]<<std::endl;
    copyToTracker(d_detector_output,tracker1.d_Z,Num_raw_detection,detection_count[0]);
    //}else{  //for each frame after 1st frame
        //prediction
        make_prediction(tracker1,trackcount);
        //make_prediction(tracker1,trackcount);
        //compute IOU and cost matrixbetween predicted box and detection
        std::cout<<"track count:"<<trackcount<<"  detection count:"<<detection_count[0]<<std::endl;
        tracker_compute_IOU(tracker1,trackcount,detection_count[0]);
        //hungarian assignment

        //update using kalman gain

        //Handle unmatched tracks

        //Create new tracks for unmatched detections

        //Output confirmed tracks
    //}
    //writeDevice2DArrayToFile(tracker1.d_F,tracker1.n,tracker1.n,"dF.txt");
    //writeDevice2DArrayToFile(tracker1.d_Q,tracker1.n,tracker1.n,"dQ.txt");
    //writeDevice2DArrayToFile(tracker1.d_R,tracker1.m,tracker1.m,"dR.txt");
    //writeDevice2DArrayToFile(tracker1.d_H,tracker1.n,tracker1.m,"dH.txt");
    //writeDevice2DArrayToFile(tracker1.d_state_predicted,tracker1.n,tracker1.Max_Tracks,"d_prediction.txt");
    writeDevice2DArrayToFile(tracker1.d_IOU,trackcount,detection_count[0],"d_IOU.txt");

    //writeDevice2DArrayToFile(tracker1.d_state_updated,tracker1.n,tracker1.Max_Tracks,"stateTest.txt");

    //writeDevice2DArrayToFile(tracker1.d_Pcov,tracker1.n*tracker1.Max_Tracks,tracker1.n,"PcovTest.txt");

    free_jpeg_from_host(image);
    free_jpeg_from_host(image2);


    free_preprocess_mem(d_preprocess_input,d_preprocess_output);
    cudaFree(d_detector_output);
    cudaFree(d_detection_buffer);
    cudaFree(d_class_id);
    cudaFree(d_detection_count);

    tracker1.freeOnDevice();
    std::cout<<"I am done"<<std::endl;
    return 0;
    } catch (const Ort::Exception& e) {
        printf("ONNX Error: %s\n", e.what());
    }
}