#include <onnxruntime_cxx_api.h>
#include "../include/sort_lib.h"
#include "../include/helper.h"

int main(int argc, char** argv){
	if (argc != 2) {std::cout<<"wrong argument\n";return 1;}

    std::string path = argv[1];

    ImageData image = load_jpeg_bgr_hwc_to_host(path);

    // Initialize Environment and Session Options
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");
    Ort::SessionOptions session_options;

    // Enable CUDA
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Optional: graph optimizations
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load the model
    Ort::Session session(env, L"yolo11n.onnx", session_options);

    // query the model in runtime for input/output from session
    size_t num_inputs = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    // get input name/shape from model
    std::vector<const char*> input_names(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        // name 
        char* name = session.GetInputName(i, allocator);
        input_names[i] = name;
        // shape
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes[i] = tensor_info.GetShape();
    }

    // Output
    size_t num_outputs = session.GetOutputCount();
    std::vector<const char*> output_names(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        char* name = session.GetOutputName(i, allocator);
        output_names[i] = name;
    }
    // get output shape from models
    Ort::TypeInfo out_info = session.GetOutputTypeInfo(0);
    auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = out_tensor_info.GetShape();


    //preprocess
    printf("image preprocess test for onnx runtime:\n");
    printf("convert and resize BGR HWC to RGB CHW with normalized value for ONNX input format\n");
    printf("test image input size width:%d * height: %d\n",image.width,image.height);

    int width = image.width;
    int height = image.height;

    int out_width = 640;
    int out_height = 640;
    printf("convert to %d * %d size\n",out_width,out_height);
    float* output = (float*)malloc(sizeof(float)*out_height*out_width*3);
    float* output_cpu = (float*)malloc(sizeof(float)*out_height*out_width*3);
    
    uint8_t* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input,sizeof(uint8_t)*width*height*3);
    cudaMalloc((void**)&d_output,sizeof(float)*out_height*out_width*3);
    cudaMemcpy(d_input,image.data,sizeof(uint8_t)*width*height*3,cudaMemcpyHostToDevice);

    
	cudaDeviceSynchronize();
    struct timespec start2;
    getstarttime(&start2);
    frame_preprocess(d_input,d_output,height,width,out_height,out_width);
    uint64_t consumed2 = get_lapsed(start2);
    printf("preprocess  conversion-used time: %" PRIu64 "\n",consumed2);

    free_jpeg_from_host(image);

    // Create input tensor from your preprocessed float array
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_blob.data(), input_blob.size(), input_shape.data(), input_shape.size()
    );

    // Run!
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get a pointer to the results
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
}