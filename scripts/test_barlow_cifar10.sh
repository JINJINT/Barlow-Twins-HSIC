#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
feature_dim=128

case=1
if [ $case -eq 1 ]; then
  feature_dim=128
  ckpt_path='0.005_128_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  feature_dim=128
  ckpt_path='0.05_128_128_cifar10_model.pth'
fi
ckpt_path='./results/'$ckpt_path

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --test-only=1 \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$bt \
    --load-ckpt=1 \
    --pretrained-path=$ckpt_path

