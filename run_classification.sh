#! /bin/sh

cd build; 

precision="1" # (0:f32, 1:f16, 2:int8)

./TFLiteClassification /home/sara/Documents/Master-thesis/TFLite/models/class_models/768-fp32.tflite $precision /home/sara/Documents/Master-thesis/TFLite/data/boat2.jpg