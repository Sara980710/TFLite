#! /bin/bash

cd build; 

precision="1" # (0:f32, 1:f16, 2:int8)

./TFLiteYolov5 /home/sara/Documents/Master-thesis/TFLite/models/object_detection/exp16/weights/epoch80-fp16.tflite $precision /home/sara/Documents/Master-thesis/TFLite/data/yolo.jpg 