ifeq ($(OS),Windows_NT)
    default: windows
else
    default: linux
endif



linux:
	# nvcc -ccbin /usr/bin/gcc-11 test_image_preprocess.cu ../preprocess_lib.cu -o preprocesstest.exe
	# nvcc -ccbin /usr/bin/g++-11 test_image_preprocess.cu ../preprocess_lib.cu -o preprocesstest.exe
	# use above if using older cuda
	nvcc test_image_preprocess.cu ../preprocess_lib.cu -o preprocesstest.exe
	
windows:
	nvcc test_image_preprocess.cu ../preprocess_lib.cu -o preprocesstest.exe
	del /Q *.exp *.lib
	
