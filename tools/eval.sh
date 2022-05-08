#!/bin/bash

#set -eux

# Usage: bash tools/train.sh <config-file> <master-port>

if [ $# == 0 ]; then
  echo "USAGE: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh <config-file> <master-port>"
  echo " e.g.1: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cfg.yaml"
  echo " e.g.2: CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cfg.yaml 16233"
  exit 1
fi

master_port="15231"
cfg_file=""
if [ $# == 1 ]; then
  cfg_file=$1
elif [ $# == 2 ]; then
  cfg_file=$1
  master_port=$2
fi

export PYTHONPATH=.

python -m torch.distributed.launch --nproc_per_node=4 --master_port="${master_port}" \
  tools/train.py -cfg "${cfg_file}" \
  --opt-level O1 \
  --evaluate
