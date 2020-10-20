#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "Path to Carrada dataset is mandatory, please add it as parameter of the script."
    exit 1
fi

echo "Path to Carrada dataset -> " $1

python ../scripts/set_path.py $1 && pytest test_instances.py && pytest test_rd_points.py && pytest test_centroid.py && pytest test_annotation_files.py
