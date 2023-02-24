#!/bin/bash

model_file="models/nltcs.pt"
data_file="datasets/nltcs/nltcs.train.data"
python3 sample.py --num_vars 16 --device cuda --cuda_core 1 --model_path $model_file --data_path $data_file --num_seeds 1 --num_samples 100 --evidence_count 1
