ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif

linux:
	# use -ccbin /usr/bin/gcc-11 for myself
	nvcc hungarian_stream.cu -o hungarian_stream.exe -lrt
windows:
	nvcc hungarian_stream.cu -o hungarian_stream.exe

