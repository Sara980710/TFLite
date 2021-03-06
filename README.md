# Master's thesis: TFLite
[Link to master's thesis repo](https://github.com/Sara980710/master_thesis)

version: [tensorflow lite 2.8.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0)

https://github.com/clbr/radeontop

## Installation of TF
I used this tutorial: [youtube](https://www.youtube.com/playlist?list=PLYV_j9XEhvorTV-ClcNA2xUb5YsdUHgRX)

### OpenCV
````bash
sudo apt update
sudo apt install libopencv-dev python3-opencv
````
### OpenGL
````bash
sudo apt update
sudo apt install ocl-icd-opencl-dev
````
(might need sudo apt-get install freeglut3-dev)

## Configure and Build using CmakeLists.txt
Run following:
````
bash configure.sh
bash build.sh
````
## Run (TF-lite model on C++)
* run_classification.sh - classify one image
* run_yolo.sh - detect boats on one image
* run_timeit.sh - time the inference
* run_timeit_multiple.sh - time the inference on multiple models at once
* run_memory.sh - measure the memory usage of the programs

## Detector
* method 0: just invoke
* method 1: convert image first to float
* method 2: convert only tile to float
* method 3: copy values one by one

## Measurements
### Time
arg: [path to model, iterations, precision]
````
./build/TFLiteTimeitGPUSimple /home/spacecloud/ebara/ml_performancetests/models/yolo_models/3072-fp16.tflite 100 1 
````
arg: [path to model, iterations, precision, path to image, threads, device, verbose]
````
./build/TFLiteTimeitInference /home/spacecloud/ebara/ml_performancetests/models/yolo_models/3072-fp16.tflite 100 1 
````
arg: [path to model, iterations, precision, path to image, threads, device, method, verbose, invoke (1 or 0)]
````
./build/TFLiteTimeit /home/spacecloud/ebara/ml_performancetests/models/yolo_models/5024-fp16.tflite 3 1 /home/spacecloud/ebara/ml_performancetests/data/big.jpg 1 1 2 0 1
````

### Memory 
Turn on/off swap-memory:
* ````sudo swapoff -a  ````
* ````sudo swapon -a  ````

In terminal 1:
````
top -b -n 1000 > iterations.log
````
In terminal 2:
arg: [path to model, precision, path to image, threads, device, method (0 or 1 or 2 or 3), verbose, detect]
````
./build/TFLiteMemory /home/spacecloud/ebara/ml_performancetests/models/yolo_models/3072-fp16.tflite 1 /home/spacecloud/ebara/ml_performancetests/data/big.jpg 1 1 2 0 1
````
or 
arg: [path to model, precision, threads, device, verbose]
````
./build/TFLiteMemoryInference /home/spacecloud/ebara/ml_performancetests/models/yolo_models/3072-fp16.tflite 1 1 1 0
````
or
arg: [path to image, convert to float or not]
````
./build/TFLiteMemoryImage /home/spacecloud/ebara/ml_performancetests/data/big.jpg 1 
````
In terminal 1:
````
cat iterations.log | grep 'MiB Mem'
cat iterations.log | grep TFLiteMemory
cat iterations.log | grep TFLite >> memory/filename.log;
````

### Python programs
````
python3 detector.py --weights models/yolo_models/1440-fp16.tflite --image data/big.jpg --gpuoff
python3 detector_slim.py --weights models/yolo_models/1440-fp16.tflite --image data/big.jpg
python3 detector_lite.py --weights models/yolo_models/1440-fp16.tflite --image data/big.jpg
````

### (Maximum resident set size)
````
/usr/bin/time -v <program> <args>
````
## Makefile
VPN and SSH is used to achieve access to the satellite computer where TFLite is already installed.
* Copy the src folder, models folder and Makefile to the satellite computer using scp (safe copy)
* run:```` make ```` to build the executable files
* 
