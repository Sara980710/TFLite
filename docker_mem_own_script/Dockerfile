FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install linux packages
RUN apt-get update && apt-get install -y zip tmux libgl1-mesa-glx 

WORKDIR /

RUN mkdir memory_test
WORKDIR /memory_test/

COPY ./TFLiteMemory .