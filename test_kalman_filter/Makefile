ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif



linux:
	# nvcc -ccbin /usr/bin/gcc-11 test_kalman_filter.cu ../kalman_filter_lib.cu -o kalman_filter -lrt -lcublas -lcusolver
	# nvcc -ccbin /usr/bin/g++-11 test_kalman_filter.cu ../kalman_filter_lib.cu -o kalman_filter -lrt -lcublas -lcusolver
	# use above if using older cuda
	nvcc test_kalman_filter.cu ../kalman_filter_lib.cu -o kalman_filter -lrt -lcublas -lcusolver
	
windows:
	nvcc test_kalman_filter.cu ../kalman_filter_lib.cu -o kalman_filter -lcublas -lcusolver
	del /Q *.exp *.lib
	
