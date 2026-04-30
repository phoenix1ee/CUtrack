# CUtrack
A fully CUDA accelerated implementation of Multi-objects tracking using SORT algorithm, with 100% GPU (On device) data pipeline to achieve near to zero host-device data copy and maximum performance on multi-stream and multi objects scenarios



## Algorithm Design
This tracker combine several core ideas:
1. Use YOLO as detector for inference and keep flexibility to swap to other models, e.g. neural network/ML driven
2. Use Auction Algorithm for matching of tracks and detections
3. Use a customised tracker object to allow runtime vector size declaration to support different state/measurement combination for compatibility with all scenarios other than object tracking in fixed frame, but also 2D/3D world, linearize/non-linearize robots kinematic models
4. Use a 100% on device data pipeline to allow for maximum efficiency, parallelism and scalability
5. Combination of custom kernels and CUBLAS, CUSOLVER library for maximum performance and flexibility

## Framework involved
### OpenCV
-for extract of frame data from video stream and output display
### ONNX runtime
-for running yolo detection models
### CUDA libraries
-CUBLAS for batched matrix processing
-CUSOLVER for Cholesky Factorization and solving linear system for Kalman Gain

## Major Components
1. Input stream/extract tensor with OpenCV
2. Frame Pre-processing
3. Tracker initialization
4. Detection with ONNX runtime and YOLO
4. NMS post-processing to suppress ghost/duplicate detections
5. State Estimation
6. Computing IOU
7. Matching of tracks and detections using Auction Algorithm
8. Update and correction of tracks with Kalman filter
9. Output and Display with OpenCV for demonstration

## Project Folder Structure

```
project_root/
├── include/                       # Public headers
│   ├── helper.h                   # header file of helper functions
│   ├── hungarian_cpu_vectorized.h # header file of a AVX accelerated CPU function for benchmarking purpose
│   ├── inference.h                # header file of a wrapper class for ONNX runtime
│   └── sort_lib.h                 # main header file of all SORT related kernel wrapper functions
├── ExportYOLOmodel.py             # python script to export YOLO model
├── hungarian_cpu_vectorized.cpp   # cpu vectorized function of row and column min reduction
├── auctionAlgo_lib.cu             # library for auction algorithm
├── input_lib.cu                   # library for handling input
├── IOU_lib.cu                     # library for IOU matrix calculation
├── kalman_filter_lib.cu           # library for Kalman filter
├── preprocess_lib.cu              # library for preprocessing
├── main.cu                        # main file, pending, with initialization test only
├── duck.mp4                       # a funny demo video for testing
├── Makefile                       #
├── README.md                      #

```

## Download and test:

### Pre-requisite before make and run

1.ONNX runtime
2.OpenCV libraries
3.CUDA

The project is developed and tested with Cuda 11.8 (My hardware is older), ONNX 1.18 and OpenCV 4.12. Make sure you have all the libraries in machine and have the path included in system variables. If running windows, in my case, you need to have the ONNX runtime dll files:\
-onnxruntime_providers_cuda.dll\
-onnxruntime_providers_shared.dll\
-onnxruntime.dll \
under the project root because windows 11 has a built-in ONNX runtime of 1.17 that cause version conflicts when compiling.

1. Clone the repo
2. Navigate to project root folder
3. run python script to export the model files(.onnx and .pt) to project root or you prepare your own models
4. revise the makefile with your own ONNX and OpenCV path
3. Run make command to compile and run the executable


### Instruction to Run:

```commandline
>>make
>>./main.exe ./duck.mp4
```