#!/usr/bin/env

# NOTE: We have not officially released our zero-shot classification pipeline yet.
# This script and its corresponding python entryfile may be changed in further releases.

# usage: bash run_scripts/imagenet_zeroshot_eval.sh CKPT_PATH

# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

DATAPATH=../imagenet-1k_val # provide the path of imagenet-val directory in torchvision dataset format
resume=${1}
vision_model=${2:-ViT-B-16}

python -u cn_clip/eval/zeroshot_evaluation.py \
    --imagenet-val="${DATAPATH}" \
    --img-batch-size=64 \
    --context-length=32 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=RoBERTa-wwm-ext-base-chinese