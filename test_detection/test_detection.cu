#include <onnxruntime_cxx_api.h>
#include "../include/sort_lib.h"
#include "../include/helper.h"

int main(int argc, char** argv){
try {
    if (argc != 2) {std::cout<<"wrong argument\n";return 1;}

    std::string path = argv[1];

    ImageData image = load_jpeg_bgr_hwc_to_host(path);

    // Initialize Environment and Session Options
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");
    //Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "YOLO_Inference");

    Ort::SessionOptions session_options;
    //session_options.SetLogSeverityLevel(0);  // 0 = VERBOSE

    // Enable CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    std::cout<<"session option for cuda set"<<std::endl;

    // Optional: graph optimizations
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load the model
    Ort::Session session(env, L"../yolo11n.onnx", session_options);
    std::cout<<"model loaded and session created"<<std::endl;

    // query the model in runtime for input/output from session
    size_t num_inputs = session.GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    // get input name/shape from model
    std::vector<const char*> input_names(num_inputs);
    std::vector<std::string> input_name_temp;
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    std::cout<<"total number of input needed:"<<num_inputs<<std::endl;
    for ( int i = 0; i < num_inputs; i++) {
        // get a copy of name , images
        auto name = session.GetInputNameAllocated(i, allocator);
        input_name_temp.push_back(std::string(name.get()));

        // shape of input, vector of int64, [1 3 640 640]
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes[i] = tensor_info.GetShape();
    }

    for ( int i = 0; i < num_inputs; i++) {
        // get the pointer of the name copies, images
        input_names[i]=input_name_temp[i].c_str();
        std::cout <<"input "<<i<<":"<< input_names[i] << std::endl; 
        // print the shape of each input
        std::cout<<"ranks of shape: "<<input_shapes[i].size()<<"-->";
        for (auto d : input_shapes[i]) {
            std::cout << d << " ";
        }
        std::cout<<std::endl;
    }

    // Output, should only have 1 output0
    size_t num_outputs = session.GetOutputCount();
    std::vector<std::vector<int64_t>> output_shapes(num_outputs);
    std::vector<int64_t> output_shapes_size(num_outputs);
    
    std::vector<const char*> output_names(num_outputs);
    //Ort::AllocatedStringPtr** output_names = (Ort::AllocatedStringPtr**)malloc(sizeof(Ort::AllocatedStringPtr*)*num_outputs);
    std::cout<<"total number of output needed:"<<num_outputs<<std::endl;
    for ( int i = 0; i < num_outputs; i++) {
        // name output0
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = name.get();
        std::cout <<"output "<<i<<":"<< output_names[i] << std::endl; 

        // get output shape from models
        Ort::TypeInfo out_info = session.GetOutputTypeInfo(i);
        auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();
        output_shapes[i] = out_tensor_info.GetShape();
        std::cout<<"ranks of shape: "<<output_shapes[i].size()<<"-->";
        int64_t temp = 1;
        for (auto d : output_shapes[i]) {
            std::cout << d << " ";
            temp*=d;
        }
        output_shapes_size[i]=temp;
        std::cout<<"total size of shape:"<<output_shapes_size[i];
        std::cout<<std::endl;
    }
    

    //preprocess
    printf("image preprocess test for onnx runtime:\n");
    printf("convert and resize BGR HWC to RGB CHW with normalized value for ONNX input format\n");
    printf("test image input size width:%d * height: %d\n",image.width,image.height);

    int width = image.width;
    int height = image.height;
    size_t total_input_data_size = sizeof(uint8_t)*width*height*3;

    int out_width = input_shapes[0][3];
    int out_height = input_shapes[0][2];
    size_t total_output_data_size = sizeof(float)*out_height*out_width*3;
    printf("convert to %d * %d size\n",out_width,out_height);
    float* output = (float*)malloc(total_output_data_size);
    
    uint8_t* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input,total_input_data_size);
    cudaMalloc((void**)&d_output,total_output_data_size);
    cudaMemcpy(d_input,image.data,total_input_data_size,cudaMemcpyHostToDevice);
    std::cout<<"finished copy image data to device"<<std::endl;
	cudaDeviceSynchronize();
    //struct timespec start2;
    //getstarttime(&start2);
    frame_preprocess(d_input,d_output,height,width,out_height,out_width);
    //uint64_t consumed2 = get_lapsed(start2);
    //printf("preprocess  conversion-used time: %" PRIu64 "\n",consumed2);

    free_jpeg_from_host(image);
    std::cout<<"finished preprocessing"<<std::endl;
    // Create input tensor from your preprocessed float array
    Ort::MemoryInfo input_memory_info("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        input_memory_info, d_output, total_output_data_size, 
        input_shapes[0].data(), input_shapes[0].size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Preallocate Output Buffer on GPU
    float* d_yolo_output;
    size_t total_yolo_output_data_size = output_shapes_size[0]*sizeof(float);
    cudaMalloc((void**)&d_yolo_output,total_yolo_output_data_size);
    Ort::MemoryInfo output_memory_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
    Ort::Value output_tensor = Ort::Value::CreateTensor(
        output_memory_info, d_yolo_output, total_yolo_output_data_size, 
        output_shapes[0].data(), output_shapes[0].size());

    //Bind session and the preallocated yolo output buffer
    //Bind the output once for it never change
    Ort::IoBinding io_binding(session);
    io_binding.BindOutput("output0", output_tensor);
    //Bind input for the test
    io_binding.BindInput("images", input_tensor);

    // Run!
    //std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    cudaDeviceSynchronize();

    // Get a pointer to the results
    //float* d_raw_output = output_tensor[0].GetTensorMutableData<float>();
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_yolo_output);
    std::cout<<"I am done"<<std::endl;
    } catch (const Ort::Exception& e) {
        printf("ONNX Error: %s\n", e.what());
    }
}