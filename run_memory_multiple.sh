#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "0" "1" ) #( "0" "1" "2" ) 
declare -a types=( "yolo_models" "class_models" ) #( "yolo_models" "class_models" )
declare -a sizes=( "192" "384" "768" "1152" "1536" ) # ( "32" "192" "384" "768" "1152" "1536" )
declare -a devices=( "0" "1" ) #( "1" "0" ) # (0:cpu or 1:gpu)
declare -a batch_sizes=( "1" ) #( "1" "5" "10" "50" "100")

threads="1"
verbose="0" # (0:false, 1:true)

image="big.jpg"

#pathTFLite="/TFLite" # Aiqu
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5

savePath="/project/tflite_gpu_measures" # Aiqu

echo "Info: iterations:$iterations, path_models:$pathTFLite/models, image: $image, nr threads: $threads, verbose: $verbose"

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

                /usr/bin/time -v ./build/TFLiteMemory $pathTFLite/models/$type/$model $precision $pathTFLite/data/$image $threads $device $verbose;
                cd ..;
                
                sleep 1;
                echo ""
                sleep 10;

                done
            done
        done
    done
done
