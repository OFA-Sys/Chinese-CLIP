#!/bin/bash

# Usage: see example script below.
# bash run_scripts/zeroshot_eval.sh 0 \
#     ${path_to_dataset} ${dataset_name} \
#     ViT-B-16 RoBERTa-wwm-ext-base-chinese \
#     ${ckpt_path}

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

path=${2}
dataset=${3}
datapath=${path}/datasets/${dataset}/test
savedir=${path}/save_predictions
vision_model=${4} # ViT-B-16
text_model=${5}
resume=${6}
label_file=${path}/datasets/${dataset}/label_cn.txt
index=${7:-}

mkdir -p ${savedir}

python -u cn_clip/eval/zeroshot_evaluation.py \
    --datapath="${datapath}" \
    --label-file=${label_file} \
    --save-dir=${savedir} \
    --dataset=${dataset} \
    --index=${index} \
    --img-batch-size=64 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model}
