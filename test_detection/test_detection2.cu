#include <onnxruntime_cxx_api.h>
#include "../include/sort_lib.h"
#include "../include/helper.h"
#include "../include/inference.h"

int main(int argc, char** argv){
try {
    if (argc != 2) {std::cout<<"wrong argument\n";return 1;}

    std::string path = argv[1];

    ImageData image = load_jpeg_bgr_hwc_to_host(path);

    YoloDetector detector;
    detector.load_model(L"../yolo11n.onnx");
    detector.extract_model_data();

    //preprocess
    printf("image preprocess test for onnx runtime:\n");
    printf("convert and resize BGR HWC to RGB CHW with normalized value for ONNX input format\n");
    printf("test image input size width:%d * height: %d\n",image.width,image.height);

    int width = image.width;
    int height = image.height;
    size_t data_size_to_preprocess = image.size;

    int out_width = detector.input_shapes[0][3];
    int out_height = detector.input_shapes[0][2];
    size_t data_size_after_preprocess = sizeof(float)*out_height*out_width*3;
    printf("convert to %d * %d size\n",out_width,out_height);
    
    uint8_t* preprocess_input;
    float* preprocess_output;
    cudaMalloc((void**)&preprocess_input,data_size_to_preprocess);
    cudaMalloc((void**)&preprocess_output,data_size_after_preprocess);

    float* detector_output;
    size_t detector_output_size = detector.output_shapes_size[0]*sizeof(float);
    cudaMalloc((void**)&detector_output,detector_output_size);

    //create the tensor for the model
    detector.create_input_tensor(preprocess_output,data_size_after_preprocess);
    detector.create_output_tensor(detector_output,detector_output_size);
    detector.bind_tensors();

    cudaMemcpy(preprocess_input,image.data,data_size_to_preprocess,cudaMemcpyHostToDevice);
    std::cout<<"finished copy image data to device"<<std::endl;
    //struct timespec start2;
    //getstarttime(&start2);
    frame_preprocess(preprocess_input,preprocess_output,height,width,out_height,out_width);
    //uint64_t consumed2 = get_lapsed(start2);
    //printf("preprocess  conversion-used time: %" PRIu64 "\n",consumed2);
    cudaDeviceSynchronize();
    free_jpeg_from_host(image);
    std::cout<<"finished preprocessing"<<std::endl;
    
    detector.run();
    //Get a pointer to the results
    //float* d_raw_output = output_tensor[0].GetTensorMutableData<float>();
    
    float* d_processed_detection;
    size_t Num_raw_detection = detector.output_shapes[0][2];//number of detections 8400
    int height_raw_detection = detector.output_shapes[0][1];//number of 0:3 coordinates, 4-83, score of each class
    cudaMalloc((void**)&d_processed_detection,Num_raw_detection*sizeof(float)*5);

    int* d_raw_class_id;
    cudaMalloc((void**)&d_raw_class_id,Num_raw_detection*sizeof(int));

    int* d_final_class_id;
    cudaMalloc((void**)&d_final_class_id,Num_raw_detection*sizeof(int));

    int* d_detection_count;
    cudaMalloc((void**)&d_detection_count,sizeof(int));

    NMS(detector_output,d_raw_class_id,Num_raw_detection,height_raw_detection,d_processed_detection,d_final_class_id,d_detection_count);
    
    float* testoutput = (float*)malloc(Num_raw_detection*sizeof(float)*5);
    cudaMemcpy(testoutput,d_processed_detection,Num_raw_detection*sizeof(float)*5,cudaMemcpyDeviceToHost);

    float* testoutputraw = (float*)malloc(Num_raw_detection*sizeof(float)*5);
    cudaMemcpy(testoutputraw,detector_output,Num_raw_detection*sizeof(float)*5,cudaMemcpyDeviceToHost);

    int* testclassoutput = (int*)malloc(Num_raw_detection*sizeof(int));
    cudaMemcpy(testclassoutput,d_final_class_id,sizeof(int)*Num_raw_detection,cudaMemcpyDeviceToHost);

    int totalcount[1]={0};
    cudaMemcpy(totalcount,d_detection_count, sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"total detections:"<<totalcount[0]<<std::endl;
    write2DArrayToFile(testoutputraw,5,Num_raw_detection,"rawdetection.txt");
    write2DArrayToFile(testoutput,5,Num_raw_detection,"sorteddetection.txt");
    write2DArrayToFileInt(testclassoutput,1,Num_raw_detection,"detectedclass.txt");

    cudaFree(preprocess_input);
    cudaFree(preprocess_output);
    cudaFree(detector_output);
    cudaFree(d_processed_detection);
    cudaFree(d_raw_class_id);
    cudaFree(d_final_class_id);
    cudaFree(d_detection_count);

    free(testoutput);
    free(testoutputraw);
    free(testclassoutput);
    std::cout<<"I am done"<<std::endl;
    } catch (const Ort::Exception& e) {
        printf("ONNX Error: %s\n", e.what());
    }
}