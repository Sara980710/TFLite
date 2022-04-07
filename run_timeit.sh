#! /bin/bash

cd build; 

precision="0" # (0:f32, 1:f16, 2:int8)
iterations="100"
model="32-fp32.tflite"
type="yolo_models"

./TFLiteTimeit /home/sara/Documents/Master-thesis/TFLite/$type/$model $iterations $precision
