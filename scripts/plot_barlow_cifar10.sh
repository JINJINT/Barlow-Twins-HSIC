#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
feature_dim=128
fig_token=''

case=2
if [ $case -eq 1 ]; then
  feature_dim=128
  lambda=0.005
  ckpt_path='0.005_128_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  feature_dim=128
  lambda=0.05
  ckpt_path='0.05_128_128_cifar10_model.pth'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$fig_token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=0 \
  python plot.py \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$bt \
    --ckpt_path=$ckpt_path \
    --fig-dir=$fig_dir


