ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif



linux:
	# nvcc -Xcompiler -mavx2 -ccbin /usr/bin/gcc-11 hungarian_algo_test.cu ../hungarian_lib.cu ../hungarian_cpu_vectorized.cpp -o hungarian_algo_test -lrt
	# nvcc -Xcompiler -mavx2 -ccbin /usr/bin/g++-11 hungarian_algo_test.cu ../hungarian_lib.cu ../hungarian_cpu_vectorized.cpp -o hungarian_algo_test -lrt
	# use above if using older cuda
	nvcc -Xcompiler -mavx2 -ccbin /usr/bin/g++-11 hungarian_algo_test.cu ../hungarian_lib.cu ../hungarian_cpu_vectorized.cpp -o hungarian_algo_test -lrt
windows:
	nvcc test_hungarian_algo.cu ../hungarian_lib.cu ../hungarian_cpu_vectorized.cpp -o test_hungarian_algo
	del /Q *.exp *.lib

