#!/bin/bash

target_dir="../src/openpi/training/maniskill_data"
array=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq)

mkdir -p "$target_dir"

for i in "${array[@]}"; do
    huggingface-cli download robotics-diffusion-transformer/maniskill-model \
        "demo_1k_part_${i}" \
        --local-dir "$target_dir"
done

cd "$target_dir" || exit 1

cat demo_1k_part_* > demo_1k.zip
