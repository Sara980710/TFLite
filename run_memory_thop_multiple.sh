#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "1" ) #( "0" "1" "2" ) 
declare -a types=( "yolo_models" ) #( "yolo_models" "class_models" )
declare -a sizes=( "288" "480" "672" ) # ( "32" "192" "384" "768" "1152" "1536" "3072")
declare -a devices=( "1" ) #( "1" "0" ) # (0:cpu or 1:gpu)
declare -a batch_sizes=( "3" "4" ) #"2" "3" "4" )

threadses=( "1" )
methods=( "3" ) #(0 or 1 or 2 or 3)

verbose="0" # (0:false, 1:true)
detect="1" # (0:false, 1:true)

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
                         
    # Get model name
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

    # Do stuff
    if [ "$threads" = "1" ] || ([ "$threads" != "1" ] && [ "$device" = "0" ]); then

        echo "--------------------------------------------------------------"
        echo "Method: $method, device:$device, Type: $type, Precision: $precision, Size: $size, Threads: $threads, Batch-size: $bs, Model: $model, Detect: $detect"
        echo "./build/TFLiteMemory $pathTFLite/models/$type/$model $precision $pathTFLite/data/$image $threads $device $method $verbose $detect;"

        echo "Starting measure-session"
        screen -S measure -d -m -L -Logfile iterations.log top &
        sleep 3;
        ./build/TFLiteMemory $pathTFLite/models/$type/$model $precision $pathTFLite/data/$image $threads $device $method $verbose $detect;

        echo "Stopping measure-session"
        screen -XS measure quit;

        echo "Saving results"
        echo "                                   VIRT    RES     SHR     %CPU   %MEM" >> memory/${model}_${type}_d${device}_m${method}_t${threads}.log;
        sleep 3;
        cat iterations.log | grep TFLite >> memory/${model}_${type}_d${device}_m${method}_t${threads}.log;
        rm iterations.log;

        sleep 3;
    else
    echo "SKIPPED Method: $method, device:$device, Type: $type, Precision: $precision, Size: $size, Threads: $threads, Batch-size: $bs, Model: $model "
    fi
done
done
done
done
done
done
done

cp screenlog.0 memory/.;
zip -r memory.zip memory;