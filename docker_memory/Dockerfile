FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install linux packages
RUN apt-get update && apt-get install -y zip tmux libgl1-mesa-glx 

# Cmake
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get update && apt-get -y install cmake
RUN apt update && apt install libopencv-dev python3-opencv ocl-icd-opencl-dev -y

WORKDIR /

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache 
RUN git clone https://github.com/Sara980710/TFLite.git

WORKDIR /TFLite/

RUN bash configure.sh
RUN bash build.sh
