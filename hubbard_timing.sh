#!/bin/bash

set -e

source ~/.venv/compare/bin/activate

data_dir=data/hubbard_timing
input_files=($(ls ${data_dir}/*.json))

for input_file in ${input_files[@]}; do
    output_file=${input_file/json/hdf5}
    # echo $input_file $output_file
    python hubbard.py $input_file $output_file
done

deactivate