#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "Path to Carrada dataset is mandatory, please add it as parameter of the script."
    exit 1
fi

echo "Path to Carrada dataset -> " $1

python set_path.py $1 && python generate_instances.py && python generate_rd_points.py && python generate_centroids.py && python generate_annotation_files.py
