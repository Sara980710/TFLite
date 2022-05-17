#! /bin/bash

cd build; 

precision="1" # (0:f32, 1:f16, 2:int8)

#./TFLiteYolov5 /home/sara/Documents/Master-thesis/TFLite/models/yolo_models/epoch80-fp16.tflite $precision /home/sara/Documents/Master-thesis/TFLite/data/yolo.jpg 
./TFLiteYolov5 /home/sara/Documents/Master-thesis/TFLite/models/yolo_models/bs_768_2-fp16.tflite $precision /home/sara/Documents/Master-thesis/TFLite/data/yolo.jpg 