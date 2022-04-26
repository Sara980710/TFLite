#! /bin/bash

#declare -a precisions=( "0" "1" "2" ) 
#declare -a types=( "yolo_models" "class_models" )
#declare -a sizes=( "32" "192" "384" "768" "1152" "1536" )

type="yolo_models"
size="1152"
precision="0" # (0:f32, 1:f16, 2:int8)

iterations="1"
image="big.jpg"
threads="1"
device="1" # (0:cpu or 1:gpu)
pathTFLite="/home/ebara/Documents/master_thesis/TFLite"


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

sleep 5;
echo "Starting program"

cd build; 
./TFLiteMemory $pathTFLite/models/$type/$model $precision $pathTFLite/data/$image $threads $device;
cd ..;

sleep 5;
echo "Killing measure-session"
tmux kill-session -t measure-session;
echo ""