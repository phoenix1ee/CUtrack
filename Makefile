ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif

linux:
	# use -ccbin /usr/bin/gcc-11 for myself
	nvcc -ccbin /usr/bin/g++-11 main.cu -o main -lrt
windows:
	nvcc main.cu -o main
	del /Q *.exp *.lib
