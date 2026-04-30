# Define all your source files in one place for cleaner build commands
SRCS = main.cu \
       preprocess_lib.cu \
       tracker_update_lib.cu \
       IOU_lib.cu \
       auctionAlgo_lib.cu \
       kalman_filter_lib.cu

ifeq ($(OS),Windows_NT)
    default: windows
    # ONNX
    INCLUDES = -I "C:/onnxruntime-win-x64-gpu-1.18.1/onnxruntime-win-x64-gpu-1.18.1/include"
    LIBS     = -L "C:/onnxruntime-win-x64-gpu-1.18.1/onnxruntime-win-x64-gpu-1.18.1/lib" -l onnxruntime
    #OpenCV Setup
    OPENCV_DIR = C:/opencv/build
    INCLUDES += -I "$(OPENCV_DIR)/include"
    # NOTE: Change 'world490' to match the actual .lib file in your OpenCV folder (e.g., 490 = v4.9.0)
    LIBS     += -L "$(OPENCV_DIR)/x64/vc16/lib" -l opencv_world4120
else
    default: linux
    # ONNX Runtime Setup
    INCLUDES = -I/usr/local/onnxruntime/include
    LIBS     = -L/usr/local/onnxruntime/lib -lonnxruntime
    
    # OpenCV Setup
    # pkg-config
    INCLUDES += $(shell pkg-config --cflags opencv4)
    LIBS     += $(shell pkg-config --libs opencv4)
endif

linux:
	# nvcc -ccbin /usr/bin/gcc-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# nvcc -ccbin /usr/bin/g++-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# use above if using older cuda
	nvcc $(SRCS) -o main $(INCLUDES) $(LIBS) -lrt -lcublas -lcusolver
	
windows:
	nvcc -Xcompiler="/std:c++17" -std=c++17 $(SRCS) $(INCLUDES) $(LIBS) -o main -lcublas -lcusolver
	del /Q *.exp *.lib
	