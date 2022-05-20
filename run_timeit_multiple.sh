#! /bin/bash

# TODO:
#-Current: batch sizes: ( "288" "480" "672" ) ( "2" "3" )

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "1" ) #( "0" "1" "2" ) 
declare -a types=( "class_models" ) #( "yolo_models" "class_models" )
declare -a sizes=( "32" "192" "480" "288" "672" "768" "1152" "1440" "2080" "3360" ) # ( "32" "192" "480" "288" "672" "768" "1152" "1440" "2080" "3360" )
declare -a devices=( "1" "0" ) #( "1" "0" ) # (0:cpu or 1:gpu)
declare -a batch_sizes=( "1" "2" "3" ) #( "1" "2" "3" "4" )

threadses=( "1" )
methods=( "3" ) #(0 or 1 or 2 or 3)

verbose="0" # (0:false, 1:true)
invoke="1" # (0:false, 1:true)
iterations="3"

image="big.jpg"

#pathTFLite="/TFLite" # Aiqu
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5
#savePath="/project/tflite_gpu_measures" # Aiqu

echo "Info: path_models:$pathTFLite/models, image: $image, verbose: $verbose"

for bs in "${batch_sizes[@]}"; do
for threads in "${threadses[@]}"; do 
for method in ${methods[@]}; do
for type in "${types[@]}"; do
for precision in "${precisions[@]}"; do
for device in ${devices[@]}; do
for size in "${sizes[@]}"; do  

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
            echo "--------------------------------------------------------------"
            echo "Iterations: $iterations, Type: $type, Precision: $precision, Size: $size, Model: $model, Batch size: $bs, nr threads: $threads, device:$device, method: $method, verbose: $verbose, invoke: $invoke"
            echo "Running: ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $method $verbose $invoke"
            ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $method $verbose $invoke

            sleep 1;
done
done
done
done
done
done
done
