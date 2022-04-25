#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128
test_bt=128

save_token=''

case=1
if [ $case -eq 1 ]; then
  lambda=0.005
  feature_dim=128
  ckpt_path='0.005_128_128_cifar10_model.pth'
elif [ $case -eq 2 ]; then
  lambda=0.05
  feature_dim=128
  ckpt_path='0.05_128_128_cifar10_model.pth'
elif [ $case -eq 3 ]; then
  lambda=0.05
  feature_dim=256
  # last ckpt
  ckpt_path='0.05_256_128_cifar10_model.pth'
  # ckpt in the middle
  ckpt_path='0.05_256_128_cifar10_model_150.pth'
elif [ $case -eq 4 ]; then
  lambda=0.05
  feature_dim=512
  ckpt_path='0.05_512_128_cifar10_model.pth'
fi
ckpt_path='./results/'$ckpt_path
fig_dir='./figs/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token
fSinVals=$fig_dir'/sinVals'
save_feats=1
fsave_feats='./saved_feats/dim'$feature_dim'_lmbda'$lambda'_bt'$bt$save_token

WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --test-only=1 \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --batch_size=$test_bt \
    --load-ckpt=1 \
    --pretrained-path=$ckpt_path \
    --fSinVals=$fSinVals \
    --save-feats=$save_feats \
    --fsave-feats=$fsave_feats

