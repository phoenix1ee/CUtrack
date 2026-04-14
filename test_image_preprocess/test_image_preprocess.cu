#include "../include/sort_lib.h"
#include "../include/helper.h"

#include <cstdint>
#include <string>
#include <stdexcept>
#include <cstring>

#include <iostream>



// CPU function to test the resize + bgr_hwc_to_rgb_chw + normalize
void preprocess_cpu(
    const uint8_t* bgr,
    int in_w,
    int in_h,
    float* out,      // RGB CHW, normalized [0,1]
    int out_w,
    int out_h,
    float pad_value = 0.0f
) {
    int HW = out_w * out_h;

    // planes: [R][G][B]
    std::fill(out + 0 * HW, out + 1 * HW, pad_value);
    std::fill(out + 1 * HW, out + 2 * HW, pad_value);
    std::fill(out + 2 * HW, out + 3 * HW, pad_value);

    float scale = std::min(
        static_cast<float>(out_w) / in_w,
        static_cast<float>(out_h) / in_h
    );

    int new_w = static_cast<int>(in_w * scale);
    int new_h = static_cast<int>(in_h * scale);

    int pad_x = (out_w - new_w) / 2;
    int pad_y = (out_h - new_h) / 2;

    auto pix = [&](int y, int x, int c) {
        return bgr[(y * in_w + x) * 3 + c];
    };

    for (int oy = 0; oy < new_h; ++oy) {
        for (int ox = 0; ox < new_w; ++ox) {
            int dst_x = ox + pad_x;
            int dst_y = oy + pad_y;
            int out_idx = dst_y * out_w + dst_x;

            float x_in = static_cast<float>(ox) / scale;
            float y_in = static_cast<float>(oy) / scale;

            int x0 = static_cast<int>(floorf(x_in));
            int y0 = static_cast<int>(floorf(y_in));
            int x1 = std::min(x0 + 1, in_w - 1);
            int y1 = std::min(y0 + 1, in_h - 1);

            float i = x_in - x0;
            float j = 1.0f - i;
            float s = y_in - y0;
            float t = 1.0f - s;

            float tj = t * j;
            float sj = s * j;
            float ti = t * i;
            float si = s * i;

            float B =
                pix(y0, x0, 0) * tj +
                pix(y1, x0, 0) * sj +
                pix(y0, x1, 0) * ti +
                pix(y1, x1, 0) * si;

            float G =
                pix(y0, x0, 1) * tj +
                pix(y1, x0, 1) * sj +
                pix(y0, x1, 1) * ti +
                pix(y1, x1, 1) * si;

            float R =
                pix(y0, x0, 2) * tj +
                pix(y1, x0, 2) * sj +
                pix(y0, x1, 2) * ti +
                pix(y1, x1, 2) * si;

            out[0 * HW + out_idx] = R / 255.0f;
            out[1 * HW + out_idx] = G / 255.0f;
            out[2 * HW + out_idx] = B / 255.0f;
        }
    }
}

//checking function
bool compare_all(const float* cpu,const float* gpu,int size,float eps = 1e-5f) {
    
    bool ok = true;
    for (int i = 0; i < size; ++i) {
        float a = cpu[i];
        float b = gpu[i];

        if (abs(a - b) > eps) {
            std::cout << "Mismatch at index " << i
                      << " CPU=" << a
                      << " GPU=" << b << "\n";
            ok = false;
            break;
        }
    }
    return ok;
}

__global__ void dummykernel(void){}

void test_preprocess(ImageData image){
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
    
    dummykernel<<<1,1>>>();
	cudaDeviceSynchronize();
    struct timespec start2;
    getstarttime(&start2);
    frame_preprocess(d_input,d_output,height,width,out_height,out_width);
    uint64_t consumed2 = get_lapsed(start2);
    printf("preprocess  conversion-used time: %" PRIu64 "\n",consumed2);

    cudaMemcpy(output,d_output,sizeof(float)*out_height*out_width*3,cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error !=cudaSuccess){
        printf("cuda memcpy failed, msg: %s", cudaGetErrorString(error));
    }
    cudaFree(d_input);
    cudaFree(d_output);

    //CPU version
    preprocess_cpu(image.data,width,height,output_cpu,out_width,out_height);

    //verify
    bool ok = compare_all(output_cpu, output, out_width*out_height*3);

    if (ok) {
        printf("Kernel output matches CPU reference\n");
    } else {
        printf("Kernel output does NOT match CPU reference\n");
    }

    free(output);
    free(output_cpu);
    delete[] image.data;
}

int main(int argc, char** argv){
	// read command line arguments
	
	if (argc != 2) {std::cout<<"wrong argument\n";return 1;}

    std::string path = argv[1];

    ImageData image = load_jpeg_bgr_hwc_to_host(path);

    test_preprocess(image);
}