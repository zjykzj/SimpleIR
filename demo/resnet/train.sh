#!/bin/bash

set -eux

project_root="/home/zj/repos/SimpleIR/"
cfg_file="demo/resnet/r18_cifar10_224_b256_e90_g4.yaml"

export PYTHONPATH="${project_root}"
echo $PYTHONPATH

cd ${project_root}

time1=$(date +%s)
echo "start time: $time1"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="22231" \
  tools/train.py -cfg ${cfg_file} \
  --opt-level O1

time2=$(date +%s)
echo "end time: $time2"

# shellcheck disable=SC2006
# shellcheck disable=SC2003
train_time=$(expr "$time2" - "$time1")
echo "train time: $train_time"