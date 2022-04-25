#!/bin/bash

dataset='cifar10'
proj_head_type='linear'

token=''

# whether to force the diag entries to be -1 (rather than 1)
corr_neg_one_on_diag=0
if [ $corr_neg_one_on_diag = 1 ]; then
  token=$token'diagNeg'
fi
loss_no_on_diag=1
if [ $loss_no_on_diag = 1 ]; then
  token=$token'_noOnDiag'
fi
loss_no_off_diag=0
if [ $loss_no_off_diag = 1 ]; then
  token=$token'_noOffDiag'
fi

bt=128

for lr in 1e-3
do
for wd in 1e-6
do
for feature_dim in 128
do
for lambda in 0.005
do

wb_name=$dataset'_'$proj_head_type'_feat'$feature_dim'_lmbda'$lambda'_lr'$lr'_wd'$wd'_bt'$bt$token

# WANDB_MODE=dryrun \
CUDA_VISIBLE_DEVICES=1 \
  python main.py \
    --dataset=$dataset \
    --corr_neg_one_on_diag=$corr_neg_one_on_diag \
    --feature_dim=$feature_dim \
    --proj-head-type=$proj_head_type \
    --lr=$lr \
    --wd=$wd \
    --batch_size=$bt \
    --loss-no-on-diag=$loss_no_on_diag \
    --loss-no-off-diag=$loss_no_off_diag \
    --lmbda=$lambda \
    --wb-name=$wb_name
done
done
done
done
