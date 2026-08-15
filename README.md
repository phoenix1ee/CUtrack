# CUtrack
- A fully CUDA accelerated implementation of Multi-objects tracker
- using SORT algorithm, with 100% GPU (On device) data pipeline 
- near to zero host-device data copy and maximum performance on multi-stream and multi objects scenarios


## Algorithm Design
This tracker combine several core ideas:
1. Use Auction Algorithm for matching tracks and detections
2. Use a customized tracker object to allow runtime vector size declaration to support different state/measurement combinations for compatibility with all scenarios other than object tracking in fixed frame, but also 2D/3D world, linearize/non-linearize robots kinematic models
3. Use a 100% on device data pipeline to allow for maximum efficiency, parallelism and scalability
4. Combination of custom kernels and CUBLAS, CUSOLVER library for maximum performance and flexibility

## Frameworks and CUDA libraries used
### OpenCV
-for extract of frame data from video stream and output display
### ONNX runtime
-for running yolo detection models for demonstration demo

### CUDA libraries
-CUBLAS for batched matrix processing
-CUSOLVER for Cholesky Factorization and solving linear system for Kalman Gain

## Major Components
1. Input stream/extract tensor with OpenCV
2. Frame Pre-processing
3. Tracker initialization
4. Detection with ONNX runtime and YOLO
5. NMS post-processing to suppress ghost/duplicate detections
6. State Estimation
7. Computing IOU
8. Matching tracks and detections using Auction Algorithm
9. Update and correction of tracks with Kalman filter
10. Output and Display with OpenCV for demonstration

## Project Folder Structure

```
project_root/
├── include/                       # Public headers
│   ├── helper.h                   # header file of helper functions
│   ├── inference.h                # header file of a wrapper class for ONNX runtime
│   └── sort_lib.h                 # main header file of all SORT related kernel wrapper functions
├── ExportYOLOmodel.py             # python script to export YOLO model
├── auctionAlgo_lib.cu             # library for auction algorithm
├── IOU_lib.cu                     # library for IOU matrix calculation
├── kalman_filter_lib.cu           # library for Kalman filter
├── preprocess_lib.cu              # library for preprocessing
├── main.cu                        # main file, pending, with initialization test only
├── test_stream_1.mp4              # a video as test stream for demo
├── Makefile                       #
├── README.md                      #

```

## Download and test:

### Pre-requisite before make and run

1.ONNX runtime\
2.OpenCV libraries\
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
>>./main.exe ./test_stream_1.mp4
```