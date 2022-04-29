#! /bin/bash


precision="0" # (0:f32, 1:f16, 2:int8)
type="yolo_models" #( "yolo_models" "class_models" )
size="768" # ( "32" "192" "384" "768" "1152" "1536" )
device="0" # (0:cpu or 1:gpu)
batch_size="2"

iterations="3"
threads="1"
verbose="1" # (0:false, 1:true)

image="medium.jpg"
#pathTFLite="/home/ebara/Documents/master_thesis/TFLite" # AI Sweden computer
pathTFLite="/home/sara/Documents/Master-thesis/TFLite" # Saras computer
#pathTFLite="/home/spacecloud/ebara/ml_performancetests" # ix5
 
if [ "$precision" = "0" ]; then
    model="${size}-fp32.tflite"
elif [ "$precision" = "1" ]; then
    model="${size}-fp16.tflite"
elif [ "$precision" = "2" ]; then
    model="${size}-int8.tflite"
fi

echo "Type: $type, Precision: $precision, Size: $size, Model: $model"
echo "Running: ./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $verbose"
./build/TFLiteTimeit $pathTFLite/models/$type/$model $iterations $precision $pathTFLite/data/$image $threads $device $verbose
