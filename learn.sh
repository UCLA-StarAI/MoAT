#!/bin/bash

twenty_datasets=("nltcs" "kdd" "plants" "baudio" "jester" \
                "bnetflix" "accidents" "tretail" "pumsb_star" "dna" \
                "kosarek" "msweb" "book" "tmovie" "cwebkb" "cr52" \
                "c20ng" "bbc" "ad" "msnbc")

my_datasets=("nltcs")

for dataset in "${my_datasets[@]}"
do
    log_file_name="logs/${dataset}.txt"
    output_model_file="models/${dataset}.pt"
    if [ -e $log_file_name ]
    then
        echo "$log_file_name exists"
    else
        echo "Training on ${dataset}"
        CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset_path datasets/ \
                --dataset $dataset --device cuda --cuda_core 1 --model MoAT --max_epoch 100 \
                --batch_size 128 --lr 0.001  \
                --log_file $log_file_name --output_model_file $output_model_file
    fi
done
