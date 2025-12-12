#!/bin/bash

set -e

# prepare the input by setting different values of l.

cd data/hubbard_timing

sizes=(2 3 4 5)
ncores=(1 2 4 6)
for size in ${sizes[@]}; do
    for nc in ${ncores[@]}; do
        jq ".l1 = ${size} | .l2 = ${size} | .n_workers = ${nc}" hubbard_in_2_2.json > hubbard_l${size}_n${nc}.json
    done
done

cd ../..

# Run the script for each value of l.

source ~/.venv/compare/bin/activate

data_dir=data/hubbard_timing
input_files=($(ls ${data_dir}/hubbard_l*.json))

for input_file in ${input_files[@]}; do
    output_file=${input_file/json/hdf5}
    # echo $input_file $output_file
    python hubbard.py $input_file $output_file
done

deactivate