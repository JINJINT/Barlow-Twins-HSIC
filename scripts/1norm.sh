#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

bt=128

for lr in 3e-4
do
for wd in 1e-6
do
for feature_dim in 512
do
for lambda in 0.005
do

wb_name='hsic_'$dataset'_'$proj_head_type'_feat'$feature_dim'_lmbda'$lambda'_lr'$lr'_wd'$wd'_bt'$bt

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python3 main.py \
    --corr_neg_one \
    --symloss \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --batch_size=$bt \
    --lmbda=$lambda \
    --wb-name=$wb_name
done
done
done
done
