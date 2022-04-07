#! /bin/bash

cd build

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "0" "1" "2" ) 
declare -a types=( "yolo_models" "class_models" )
declare -a sizes=( "32" "192" "384" "768" "1152" "1536" )

iterations="100"
device="CPU" #CPU or GPU


for type in "${types[@]}"; do
    for precision in "${precisions[@]}"; do
        for size in "${sizes[@]}"; do  
            if [ "$precision" = "0" ]; then
                model="${size}-fp32.tflite"
            elif [ "$precision" = "1" ]; then
                model="${size}-fp16.tflite"
            elif [ "$precision" = "2" ]; then
                model="${size}-int8.tflite"
            fi
            echo "Type: $type, Precision: $precision, Size: $size, Model: $model"
            if [ "$device" = "CPU" ]; then
                ./TFLiteTimeit /home/sara/Documents/Master-thesis/TFLite/$type/$model $iterations $precision
            fi
        done
    done
done