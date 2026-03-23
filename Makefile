ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif

linux:
	# use -ccbin /usr/bin/gcc-11 for myself
	nvcc -ccbin /usr/bin/g++-11 kalman_filter.cu -o kalman_filter -lrt -lcublas -lcusolver
windows:
	nvcc hungarian_stream.cu -o hungarian_stream.exe

