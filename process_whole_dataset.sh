#!/usr/bin/env bash

classes=( "chairs" "lamps" "planes" "sofas" "tables" )
for cls in "${classes[@]}"; do
    echo "Processing ${cls}"
    python preprocess_data.py \
        --data_dir data/ShapeNetCore.v2-DeepSDF/ \
        --source data/ShapeNetCore.v2/ \
        --name ShapeNetV2 \
        --split examples/splits/sv2_${cls}_train.json \
        --skip
    python preprocess_data.py \
        --data_dir data/ShapeNetCore.v2-DeepSDF/ \
        --source data/ShapeNetCore.v2/ \
        --name ShapeNetV2 \
        --split examples/splits/sv2_${cls}_test.json \
        --test \
        --skip
    python preprocess_data.py \
        --data_dir data/ShapeNetCore.v2-DeepSDF/ \
        --source data/ShapeNetCore.v2/ \
        --name ShapeNetV2 \
        --split examples/splits/sv2_${cls}_test.json \
        --surface \
        --skip
done;
