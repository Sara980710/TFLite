#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "1" ) #( "0" "1" "2" ) 
declare -a types=( "yolo_models" ) #( "yolo_models" "class_models" )
declare -a sizes=( "480" "288" "672" ) # ( "32" "192" "384" "768" "1152" "1536" "3072")
declare -a devices=( "1" ) #( "1" "0" ) # (0:cpu or 1:gpu)
declare -a batch_sizes=( "1" )

iterations="3"

threadses=( "1" )
verbose="0" # (0:false, 1:true)
method=3 #(0 or 1 or 2 or 3)

image="big.jpg"
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5

echo "Info: precisions:$precisions, types:$types, sizes:$sizes, iterations:$iterations, device:$device, path_models:$pathTFLite/models, image: $image, verbose: $verbose"

for device in ${devices[@]}; do
    for type in "${types[@]}"; do
        for precision in "${precisions[@]}"; do
            for size in "${sizes[@]}"; do  
                for threads in "${threadses[@]}"; do 
                    for bs in "${batch_sizes[@]}"; do 

                        if [ "$precision" = "0" ]; then
                            if [ "$bs" = "1" ]; then
                                model="${size}-fp32.tflite"
                            else
                                model="${size}_${bs}-fp32.tflite"
                            fi
                        elif [ "$precision" = "1" ]; then
                            if [ "$bs" = "1" ]; then
                                model="${size}-fp16.tflite"
                            else
                                model="${size}_${bs}-fp16.tflite"
                            fi
                        elif [ "$precision" = "2" ]; then
                            model="${size}-int8.tflite"
                        fi
                            echo "Type: $type, Precision: $precision, Size: $size, Model: $model, Batch size: $bs, nr threads: $threads"
                            echo "Running: ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $method $verbose"
                            ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $method $verbose

                            sleep 1;
                    done
                done
            done
        done
    done
done
