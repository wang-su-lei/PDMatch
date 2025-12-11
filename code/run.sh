#!/bin/bash    "9323" "6538" "7426" 
seeds=("7900" ) 
for seed in "${seeds[@]}"; do
    python train.py \
        --seed "$seed" \
        --labeled_ratio 0.001\
	--consistency_start 20\
        --batch_size  48\
        --labeled_bs 12\
        --exp Train \
        --gpu 0,1 \
        --epochs 50 \
        2>> stderr.log
done