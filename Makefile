#all: tflite_test tflite_test_inference tflite_test_memory tflite_test_memory_inference

#tflite_test_simple: src/tflite_timeit.cpp src/Detector.cpp src/Detector.h
#	g++ -o build/TFLiteTimeitGPUSimple src/tflite_timeit_gpu.cpp -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL

TIME = true
TIMEINFERENCE = false
MEMORY = true
MEMORYIMAGE = true
MEMORYINFERENCE = true

# timing
ifeq ($(TIME), true)
tflite_test_time: src/tflite_timeit.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteTimeit src/tflite_timeit.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 
endif
ifeq ($(TIMEINFERENCE), true)
tflite_test_time_inference: src/tflite_timeit_inference.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteTimeitInference src/tflite_timeit_inference.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 
endif

# memory
ifeq ($(MEMORY), true)
tflite_test_memory: src/tflite_mem.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteMemory src/tflite_mem.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 
endif
ifeq ($(MEMORYIMAGE), true)
tflite_test_memory_image: src/tflite_mem_image.cpp
	g++ -o build/TFLiteMemoryImage src/tflite_mem_image.cpp -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core 
endif
ifeq ($(MEMORYINFERENCE), true)
tflite_test_memory_inference: src/tflite_mem_inference.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteMemoryInference src/tflite_mem_inference.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 
endif
# other
clean:
	rm TFLiteTimeit TFLiteTimeitInference TFLiteMemory TFLiteMemoryInference|| true