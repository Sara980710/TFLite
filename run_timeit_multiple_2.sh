#! /bin/bash

# (0:f32, 1:f16, 2:int8)
declare -a precisions=( "$pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads" "1" ) #( "0" "1" "2" ) 
declare -a devices=( "0" "1" ) #( "1" "0" ) # (0:cpu or 1:gpu)

iterations="3"

threads="1"
verbose="0" # (0:false, 1:true)

image="big.jpg"
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
#pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5

echo "Info: iterations:$iterations, image: $image, nr threads: $threads, verbose: $verbose"

for device in ${devices[@]}; do
    for setting in ${settings[@]}; do

        echo "device: $device, Setting: $setting"
        echo "Running: ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $verbose"
        ./build/TFLiteTimeit $setting $device $verbose

        sleep 1;
    done
done
