#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "0" "1" ) #( "0" "1" "2" ) 
declare -a types=( "yolo_models" "class_models" ) #( "yolo_models" "class_models" )
declare -a sizes=( "1152" "1440" "1536" "3360") # ( "32" "192" "384" "768" "1152" "1536" )
declare -a devices=( "0" "1" ) #( "1" "0" ) # (0:cpu or 1:gpu)

threads="1"
verbose="0" # (0:false, 1:true)

#pathTFLite="/TFLite" # Aiqu
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5

echo "Info: path_models:$pathTFLite/models, nr threads: $threads, verbose: $verbose"

for device in ${devices[@]}; do
    for type in "${types[@]}"; do
        for precision in "${precisions[@]}"; do
            for size in "${sizes[@]}"; do  
                for bs in "${batch_sizes[@]}"; do 

                if [ "$precision" = "0" ]; then
                    model="${size}-fp32.tflite"
                elif [ "$precision" = "1" ]; then
                    model="${size}-fp16.tflite"
                elif [ "$precision" = "2" ]; then
                    model="${size}-int8.tflite"
                fi
                echo "--------------------------------------------------------------"
                echo "Type: $type, Precision: $precision, Size: $size, Model: $model, device:$device"
                echo ""
                echo "Starting program"

                /usr/bin/time -v ./build/TFLiteMemory $pathTFLite/models/$type/$model $precision $threads $device $verbose;
                cd ..;
                echo ""
                sleep 10;

                done
            done
        done
    done
done
