#!/bin/bash

set -e

IMAGE_NAME="sara980710/tflite_memory_script:v1.7"
#IMAGE_NAME="sara980710/yolov5_testkd_env:v1.0"

docker build -f Dockerfile -t $IMAGE_NAME . 
