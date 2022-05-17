#!/bin/bash

set -e

IMAGE_NAME="sara980710/tflite_memory_script:v1.2"
#IMAGE_NAME="sara980710/yolov5_testkd_env:v1.0"

docker build -f docker/Dockerfile -t $IMAGE_NAME . 
