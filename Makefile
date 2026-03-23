ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif

linux:
	# use -ccbin /usr/bin/gcc-11 for myself
	# nvcc -ccbin /usr/bin/g++-11 transposetest.cu -o transposetest -lrt -lcublas
	# nvcc -Xcompiler -mavx2 -ccbin /usr/bin/g++-11 hungarian_algo_test.cu hungarian_lib.cu hungarian_cpu_vectorized.cpp -o hungarian_algo_test -lrt
	nvcc -ccbin /usr/bin/g++-11 kalman_filter.cu -o kalman_filter -lrt -lcublas -lcusolver
windows:
	nvcc hungarian_stream.cu -o hungarian_stream.exe

