#!/bin/bash

target_dir="PATH/TO/DATASET" # Where you would like to store the downloaded files
array=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq)

mkdir -p "$target_dir"

for i in "${array[@]}"; do
    huggingface-cli download robotics-diffusion-transformer/maniskill-model \
        "demo_1k_part_${i}" \
        --local-dir "$target_dir"
done

# Then follow remaining instructions at https://huggingface.co/robotics-diffusion-transformer/maniskill-model
