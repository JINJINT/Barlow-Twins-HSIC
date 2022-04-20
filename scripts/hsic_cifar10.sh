#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

corr_neg_one=1

for lr in 1e-3
do
for wd in 1e-6
do
for feature_dim in 128 256
do
for lambda in 0.05
do

wb_name='HSIC_'$dataset'_'$proj_head_type'_feat'$feature_dim'_lmbda'$lambda'_lr'$lr'_wd'$wd

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --corr_neg_one=$corr_neg_one \
    --dataset=$dataset \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --lmbda=$lambda \
    --wb-name=$wb_name
done
done
done
done
