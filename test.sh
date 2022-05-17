#! /bin/bash

screen -S measure -d -m -L -Logfile iterations.log top &
./build/TFLiteMemory /home/sara/Documents/Master-thesis/TFLite/models/yolo_models/768-fp16.tflite 1 /home/sara/Documents/Master-thesis/TFLite/data/big.jpg 3 0 2 1;
screen -XS measure quit;
echo "                                   VIRT    RES     SHR     %CPU   %MEM" >> testing.txt;
cat iterations.log | grep TFLite >> testing.txt;
rm iterations.log;