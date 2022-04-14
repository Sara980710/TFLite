FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install linux packages
RUN apt update && apt install -y zip tmux libgl1-mesa-glx 

WORKDIR /
RUN mkdir benchmark
WORKDIR /benchmark/
COPY start.sh .

# (workaround for skipping cache):
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache 
# Clone from github
RUN git clone https://github.com/Sara980710/TFLite


