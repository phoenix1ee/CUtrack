ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif



linux:
	# nvcc -ccbin /usr/bin/gcc-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# nvcc -ccbin /usr/bin/g++-11 test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	# use above if using older cuda
	nvcc test_IOU.cu ../IOU_lib.cu -o testIOU -lrt
	
windows:
	nvcc test_IOU.cu ../IOU_lib.cu -o testIOU
	del /Q *.exp *.lib
	
