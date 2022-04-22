#! /bin/bash

cd build; 

precision="0" # (0:f32, 1:f16, 2:int8)
iterations="10"
model="768-fp32.tflite"
type="yolo_models"
image="big.jpg"

./TFLiteTimeit /home/sara/Documents/Master-thesis/TFLite/models/$type/$model $iterations $precision /home/sara/Documents/Master-thesis/TFLite/data/$image
