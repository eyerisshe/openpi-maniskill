#!/bin/bash

names=(PickCube-v1 PushCube-v1 StackCube-v1 PlugCharger-v1 PegInsertionSide-v1 PushCube-v2)

for i in "${names[@]}"; do
    DIR="../src/openpi/training/maniskill_data/demo_1k/${i}/motionplanning"
    
    # Find the .h5 file (assuming only one)
    H5_FILE=$(ls "$DIR"/*.h5 2>/dev/null)
    
    if [ -n "$H5_FILE" ]; then
        mv "$H5_FILE" "$DIR/${i}.h5"
        echo "Renamed $H5_FILE to ${i}.h5"
    else
        echo "No .h5 file found in $DIR"
    fi
done
