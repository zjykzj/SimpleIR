#!/bin/bash

#set -eux

# Usage: bash tools/bash_train.sh <config-file> <master-port>

if [ $# == 0 ]; then
  echo "USAGE: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh <config-file> <gpus> <master-port>"
  echo " e.g.1: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cfg.yaml"
  echo " e.g.2: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cfg.yaml 4"
  echo " e.g.2: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cfg.yaml 4 16233"
  exit 1
fi

cfg_file=""
gpus=4
master_port="15231"
if [ $# == 1 ]; then
  cfg_file=$1
elif [ $# == 2 ]; then
  cfg_file=$1
  gpus=$2
elif [ $# == 3 ]; then
  cfg_file=$1
  gpus=$2
  master_port=$3
fi

export PYTHONPATH=.

python -m torch.distributed.launch --nproc_per_node=$gpus --master_port="${master_port}" \
  tools/train.py -cfg "${cfg_file}"
