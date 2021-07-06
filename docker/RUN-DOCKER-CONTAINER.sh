#!/bin/bash

# Display GUI through X Server by granting full access to any external client.
xhost +

ROOT=$(cd $(dirname $0); cd ../; pwd)

nvidia-docker run --gpus all -it --net=host --ipc=host --name tmats_deepsdf -v /tmp/.X11-unix:/tmp/.X11-unix:rw  --privileged -v $ROOT:/workspace/DeepSDF -v $HOME/dataset:/workspace/dataset tmats/deepsdf:latest bash

