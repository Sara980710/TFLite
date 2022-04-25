# Master's thesis: TFLite
[Link to master's thesis repo](https://github.com/Sara980710/master_thesis)

version: [tensorflow lite 2.8.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0)

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

## Makefile
VPN and SSH is used to achieve access to the satellite computer where TFLite is already installed.
* Copy the src folder, models folder and Makefile to the satellite computer using scp (safe copy)
* run:```` make ```` to build the executable files
* 
