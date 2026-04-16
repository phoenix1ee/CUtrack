#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

class YoloDetector {
private:
    //Ort::Env env;
    //std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    //the actual strings for the input/output names
    std::vector<std::string> input_names_strs;
    std::vector<std::string> output_names_strs;

    //raw pointers for the string of the names for ONNX Runtime.
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    Ort::Value input_tensor;
    Ort::Value output_tensor;
    Ort::IoBinding io_binding;

public:
    //shapes of input/output
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> output_shapes_size;
    YoloDetector():env{nullptr},session{nullptr},input_tensor{nullptr},output_tensor{nullptr},io_binding{nullptr}{}
    //YoloDetector(): 
    //    env(ORT_LOGGING_LEVEL_WARNING, "YoloWrapper"){}
    void load_model(const fs::path& model_path){
        // take a path to model and return a session
        std::cout << "Loading path: " << model_path.string() << std::endl;
        
        if (fs::exists(model_path)) {
            try {
            //Ort::Env env2(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");
            //env = &env2;
            env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");
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
            //session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            session = Ort::Session(env, model_path.c_str(), session_options);
            std::cout<<"model loaded and session created from"<<model_path.string().c_str()<<std::endl;
            } 
            catch (const Ort::Exception& e) {
                printf("Error creating session: %s\n", e.what());
            }
        }
    }
    void extract_model_data() {
        // query the model in runtime for input/output from session
        // input
        size_t num_inputs = session.GetInputCount();
        // get input name/shape from model
        std::cout<<"total number of input needed:"<<num_inputs<<std::endl;
        for (size_t i = 0; i < num_inputs; i++) {
            // Get Name (and keep it alive in our string vector)
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names_strs.push_back(std::string(name.get()));

            // Get Shape, for Yolo11, should be vector of int64, [1 3 640 640] for input 0
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes.push_back(tensor_info.GetShape());
        }
        for ( int i = 0; i < num_inputs; i++) {
            // get the pointer of the name copies, images
            input_names.push_back(input_names_strs[i].c_str());
            std::cout <<"input "<<i<<":"<< input_names[i] << std::endl; 
            // print the shape of each input
            std::cout<<"ranks of shape: "<<input_shapes[i].size()<<"-->";
            for (auto d : input_shapes[i]) {
                std::cout << d << " ";
            }
            std::cout<<std::endl;
        }

        // output
        size_t num_outputs = session.GetOutputCount();
        // get output name/shape from model
        std::cout<<"total number of output:"<<num_outputs<<std::endl;
        for ( int i = 0; i < num_outputs; i++) {
            // name output0
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names_strs.push_back(std::string(name.get()));

            // get output shape from models
            Ort::TypeInfo out_info = session.GetOutputTypeInfo(i);
            auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();
            output_shapes.push_back(out_tensor_info.GetShape());
        }
        for ( int i = 0; i < num_outputs; i++) {
            // get the pointer of the name copies, output0
            output_names.push_back(output_names_strs[i].c_str());
            std::cout <<"output "<<i<<":"<< output_names[i] << std::endl; 
            // print the shape of each output
            std::cout<<"ranks of shape: "<<output_shapes[i].size()<<"-->";
            int64_t temp = 1;
            for (auto d : output_shapes[i]) {
                std::cout << d << " ";
                temp*=d;
            }
            output_shapes_size.push_back(temp);
            std::cout<<"total size of shape:"<<output_shapes_size[i];
            std::cout<<std::endl;
        }
    }
    void create_input_tensor(float* d_input_buffer, size_t total_data_size){
        // Create input tensor with a input buffer on device
        Ort::MemoryInfo input_memory_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        input_tensor = Ort::Value::CreateTensor(
            input_memory_info, d_input_buffer, total_data_size, 
            input_shapes[0].data(), input_shapes[0].size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    void create_output_tensor(float* d_output_buffer, size_t total_data_size){
        // Create output tensor with a output buffer on device
        Ort::MemoryInfo output_memory_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        output_tensor = Ort::Value::CreateTensor(
            output_memory_info, d_output_buffer, total_data_size, 
            output_shapes[0].data(), output_shapes[0].size());
    }
    void bind_tensors(void){
        io_binding = Ort::IoBinding(session);
        io_binding.BindInput(input_names[0], input_tensor);
        io_binding.BindOutput(output_names[0], output_tensor);
    }
    void run(void){
        session.Run(Ort::RunOptions{nullptr}, io_binding);
    }
};
