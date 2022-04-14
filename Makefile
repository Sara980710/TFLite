all: tflite_test tflite_test_gpu tflite_test_memory tflite_test_gpu_memory

tflite_test: src/tflite_timeit.cpp src/Detector.h 
	g++ -std=c++11 -O2 -march=native src/tflite_timeit.cpp -ltensorflowlite -o TFLiteTimeit

tflite_test_gpu: src/tflite_timeit_gpu.cpp src/Detector.h 
	g++ -std=c++11 -O2 -march=native src/tflite_timeit_gpu.cpp -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL -o TFLiteTimeitGPU

tflite_test_memory: src/tflite_mem.cpp src/Detector.h 
	g++ -std=c++11 -O2 -march=native src/tflite_timeit.cpp -ltensorflowlite -o TFLiteMemory

tflite_test_gpu_memory: src/tflite_mem_gpu.cpp src/Detector.h 
	g++ -std=c++11 -O2 -march=native src/tflite_timeit_gpu.cpp -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL -o TFLiteMemoryGPU

clean:
	rm TFLiteTimeit TFLiteTimeitGPU TFLiteMemory TFLiteMemoryGPU || true
