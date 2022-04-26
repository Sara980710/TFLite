all: tflite_test tflite_test_memory

tflite_test: src/tflite_timeit.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteTimeit src/tflite_timeit.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 

tflite_test_memory: src/tflite_mem.cpp src/Detector.cpp src/Detector.h
	g++ -o build/TFLiteMemory src/tflite_mem.cpp src/Detector.cpp src/Detector.h -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL 

clean:
	rm TFLiteTimeit TFLiteMemory || true