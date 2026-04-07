# CUtrack
A fully CUDA accelerated implementation of Multi-objects tracking using SORT algorithm, with 100% GPU (On device) data pipeline to archieve near to zero host-device data copy and maximum performance on multi-stream and multi objects scenarios



## Algorithm Design
This tracker combine several core ideas:
1. Use YOLO as detector and keep flexibility to swap to other models, e.g. neural network/ML driven
2. Use Hungarian Algorithm for matching
3. Use a customised tracker object to allow runtime vector size declaration to support different state/measurement combination for compatibility with all scenarios other than object tracking in fixed frame, but also 2D/3D world, linearize/non-linearize robots kinematic models
4. Use a 100% on device data pipeline to allow for maximum efficiency, parallelism and scalability
5. Use of custom kernels for beyond CUDA library performance

## Progress
Major parts:
1. Input stream/extract tensor (Pending)
2. Frame Pre-processing (Done)
3. Tracker initialization (Done)
4. Detection with ONNX runtime and YOLO(Pending)
5. State Estimation with Kalman filter (Pending)
6. Computing IOU (Done)
7. Matching of tracks and detection using Hungarian Algorithm (partially done)
8. Update of tracks with Kalman filter (partially done)

## Project Folder Structure

```
project_root/
├── include/                       # Public headers
│   ├── helper.h                   # header of helper function
│   ├── hungarian_cpu_vectorized.h # header of a AVX accelerated CPU function for benchmarking purpose
│   └── sort_lib.h                 # main header of all SORT related kernel wrapper functions
├── test_hungarian_algo/           # folder for test of hungarian algorithm function
│   ├── test_hungarian_algo.cu     # test file
│   └── Makefile                   #
├── test_image_preprocess/         # folder for test of image preprocessing for YOLO model
│   ├── stb_image.h                # library for importing JPEG for test only
│   ├── test_image_preprocess.cu   # test file
│   ├── test_input.JPG             # sample JPG for test
│   └── Makefile                   #
├── test_IOU/                      # folder for test of IOU matrix calculation
│   ├── test_IOU.cu                # test file
│   └── Makefile                   #
├── test_kalman_filter/            # folder for test of kalman filter calculation
│   ├── test_kalman_filter.cu      # test file
│   └── Makefile                   #
├── ExportYOLOmodel.py             # python script to export YOLO model
├── hungarian_cpu_vectorized.cpp   # cpu vectorized function of row and column min reduction
├── hungarian_lib.cu               # library for hungarian algorithm
├── IOU_lib.cu                     # library for IOU matrix calculation
├── kalman_filter_lib.cu           # library for Kalman filter
├── preprocess_lib.cu              # library for preprocessing
├── main.cu                        # main file, pending, with initialization test only
├── Makefile                       #
├── README.md                      #

```
## Available Test
1. Test of image pre-process

2. Test of hungarian algorithm

3. Test of IOU computation

4. Test of Kalman filter

5. Testing of main, for tracker initialization only.

## How to download and run:

1. Clone the repo
2. Navigate to test folders with test_*, e.g. test_IOU
3. Run make command


### Instruction to Run:

```commandline
#test the IOU function
>>make

>>./testIOU
```