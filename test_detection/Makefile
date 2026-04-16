ifeq ($(OS),Windows_NT)
    default: windows
    INCLUDES = -I "C:/onnxruntime-win-x64-gpu-1.18.1/onnxruntime-win-x64-gpu-1.18.1/include"
    LIBS     = -L "C:/onnxruntime-win-x64-gpu-1.18.1/onnxruntime-win-x64-gpu-1.18.1/lib" -l onnxruntime
else
    default: linux
    # INCLUDES = -I /usr/local/onnxruntime/include
    # LIBS     = -L /usr/local/onnxruntime/lib -l onnxruntime
endif



linux:
	# nvcc -ccbin /usr/bin/gcc-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# nvcc -ccbin /usr/bin/g++-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# use above if using older cuda
	nvcc $(INCLUDES) $(LIBS) test_detection2.cu ../input_lib.cu ../preprocess_lib.cu -o test_detection2 -lrt
	
windows:
	nvcc -Xcompiler="/std:c++17" -std=c++17 $(INCLUDES) $(LIBS) test_detection2.cu ../input_lib.cu ../preprocess_lib.cu -o test_detection2 
	del /Q *.exp *.lib
	