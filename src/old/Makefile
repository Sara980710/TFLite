

all: tflite_test

tflite_test: old/tflite_timeit_gpu.cpp
	g++ -std=c++11 -O2 -march=native old/tflite_timeit_gpu.cpp -I/opt/opencv-4.5.4/include/opencv4 -L/opt/opencv-4.5.4/lib -lopencv_gapi -lopencv_stitching -lopencv_highgui -lopencv_videoio -lopencv_ml -lopencv_video -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_photo -lopencv_imgproc -lopencv_core -ltensorflowlite -ltensorflowlite_gpu_delegate -lGLESv2 -lEGL -o TFLiteTimeit

clean:
	rm TFLiteTimeit || true