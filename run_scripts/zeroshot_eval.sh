#!/bin/bash

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

path=${2}
dataset=${3}
datapath=${path}/${dataset}/test
savedir=${path}/save_predictions/
vision_model=${4} # ViT-B-16
resume=${5}
label_file=${path}/${dataset}/label_cn.txt
index=${6:-}

python -u src/eval/zeroshot_evaluation.py \
    --datapath="${datapath}" \
    --label-file=${label_file} \
    --save-dir=${savedir} \
    --dataset=${dataset} \
    --index=${index} \
    --img-batch-size=64 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=RoBERTa-wwm-ext-base-chinese
