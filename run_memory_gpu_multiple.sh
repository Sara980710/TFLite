#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "0" "1" "2" ) 
declare -a types=( "yolo_models" "class_models" )
declare -a sizes=( "32" "192" "384" "768" "1152" "1536" )

iterations="100"
device="1" # (0:"cpu" or 1:"gpu")
threads="1"

image="big.jpg"
pathTFLite="/home/ebara/Documents/master_thesis/TFLite"
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite"
#pathTFLite=""

echo "Info: precisions:$precisions, types:$types, sizes:$sizes, iterations:$iterations, device:$device, path_models:$pathTFLite/models, image: $image, nr threads: $threads"
echo ""

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
            echo "--------------------------------------------------------------"
            echo "Type: $type, Precision: $precision, Size: $size, Model: $model"
            echo ""
            echo "Starting measure-session"
            tmux new-session -d -s measure-session "while true; 
                do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_utilization_test/$type$model.log; sleep 0.1; 
                done" &

            sleep 1;
            echo "Starting program"
            
            cd build; 
            ./TFLiteMemory $pathTFLite/models/$type/$model $precision $pathTFLite/data/$image $threads $device;
            cd ..;
            
            sleep 1;
            echo "Killing measure-session"
            tmux kill-session -t measure-session;
            echo ""
            sleep 10;
        done
    done
done