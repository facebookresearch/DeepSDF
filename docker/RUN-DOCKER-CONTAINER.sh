#!/bin/bash

# Display GUI through X Server by granting full access to any external client.
xhost +

ROOT=$(cd $(dirname $0); cd ../; pwd)

# Set DATASET_DIR if you want mount from other place than DeepSDF/data
# The directory should be like $DATA_DIR/DeepSDF/data
# The dataset directory will mount to /workspace/dataset/DeepSDF/data in the container
if [ -z $DATASET_DIR ]; then
    DATASET_DIR=$(cd $ROOT; cd ../; pwd)
fi
echo "Data mounted from ${DATASET_DIR}"

nvidia-docker run --gpus all -it --net=host --ipc=host --privileged -e DISPLAY=$DISPLAY --name tmats_deepsdf \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $ROOT:/workspace/DeepSDF \
    -v $DATASET_DIR:/workspace/dataset \
    tmats/deepsdf:latest bash

