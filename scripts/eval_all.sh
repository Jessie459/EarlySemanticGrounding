#!/bin/bash
# Evaluate one model on all RefCOCO/+/g splits, sharding every split over the given GPUs.
# Usage: bash scripts/eval_all.sh <step1x|flux2> <gpu ids, e.g. 0,1,2> [extra evaluate.py args]
set -e

model=$1
IFS=',' read -ra gpus <<< "$2"
shift 2
num_shards=${#gpus[@]}

splits=("refcoco val" "refcoco testA" "refcoco testB" "refcoco+ val" "refcoco+ testA" "refcoco+ testB" "refcocog_u val" "refcocog_u test")

for shard_id in "${!gpus[@]}"; do
    (
        for item in "${splits[@]}"; do
            read -r dataset split <<< "$item"
            CUDA_VISIBLE_DEVICES=${gpus[$shard_id]} python evaluate.py --model $model --dataset $dataset --split $split \
                --num_shards $num_shards --shard_id $shard_id "$@"
        done
    ) &
done
wait

for item in "${splits[@]}"; do
    read -r dataset split <<< "$item"
    python evaluate.py --model $model --dataset $dataset --split $split --num_shards $num_shards --summarize "$@"
done
