#!/bin/bash

cd /workspace/DeepSDF
mkdir build \
    && cd build \
    && cmake .. \
    && make -j

