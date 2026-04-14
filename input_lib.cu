#include "include/sort_lib.h"
#include "include/helper.h"

#define STB_IMAGE_IMPLEMENTATION
#include "./include/third_party/stb_image.h"

// Function to load a JPEG for testing the bgr_hwc_to_rgb_chw conversion
// Caller must free with delete[]
ImageData load_jpeg_bgr_hwc_to_host(const std::string& path) {
    int w, h, channels;

    // Force 3 channels (RGB)
    uint8_t* rgb = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!rgb) {
        throw std::runtime_error("Failed to load JPEG: " + path);
    }

    size_t size = static_cast<size_t>(w) * h * 3;
    uint8_t* bgr = new uint8_t[size];

    // Convert RGB → BGR (HWC layout preserved)
    for (size_t i = 0; i < size; i += 3) {
        bgr[i + 0] = rgb[i + 2]; // B
        bgr[i + 1] = rgb[i + 1]; // G
        bgr[i + 2] = rgb[i + 0]; // R
    }

    stbi_image_free(rgb);

    return { bgr, size, w, h };
}

// Function to load a JPEG rgb_hwc format
// Caller must free with delete[]
ImageData load_jpeg_rgb_hwc_to_host(const std::string& path) {
    int w, h, channels;

    // Force 3 channels (RGB)
    uint8_t* rgb = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!rgb) {
        throw std::runtime_error("Failed to load JPEG: " + path);
    }

    size_t size = static_cast<size_t>(w) * h * 3;

    return { rgb, size, w, h };
}

void free_jpeg_from_host(ImageData image){
    stbi_image_free(image.data);
}