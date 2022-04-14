FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install linux packages
RUN apt-get update && apt-get install -y zip tmux libgl1-mesa-glx 

# Cmake
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get update && apt-get -y install cmake

WORKDIR /

RUN git clone https://github.com/Sara980710/TFLite.git

WORKDIR /TFLite/

RUN make 