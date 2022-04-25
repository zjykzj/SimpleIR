#!/bin/bash

set -eux

# Usage: bash tools/train.sh <config-file path>

echo "Execution file name: $0"
echo "The first param: $1"

export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port="15231" \
  tools/train/train.py -cfg $1 \
  --opt-level O1