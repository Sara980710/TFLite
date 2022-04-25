#! /bin/bash

#cd build
testtype="time_measure"
#testtype="memory_measure"

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "0" "1" "2" ) 
declare -a types=("yolo_models" "class_models")
declare -a sizes=( "32" "192" "384" "768" "1152" "1536" )

iterations="100"
device="CPU" #CPU or GPU

image="big.jpg"
path_models="models"
#path_models="/home/sara/Documents/Master-thesis/TFLite/models"

echo "Info: precisions:$precisions, types:$types, sizes:$sizes, iterations:$iterations, device:$device, path_models:$path_models, image: $image, nr threads: $threads"


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

                if [ "$testtype" = "time_measure" ]; then
                    ./TFLiteTimeit $path_models/$type/$model $iterations $precision $image $threads $device
                elif [ "$testtype" = "memory_measure" ]; then
                    ./TFLiteMemory $path_models/$type/$model $iterations $precision $image $threads $device
                fi
            fi
        done
    done
done
