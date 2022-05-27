#!/bin/bash

#set -eux

# Usage: bash tools/eval.sh <config-file>

if [ $# == 0 ]; then
  echo "USAGE: CUDA_VISIBLE_DEVICES=0 bash tools/eval.sh <config-file>"
  echo " e.g.: CUDA_VISIBLE_DEVICES=0 bash tools/eval.sh configs/cfg.yaml"
  exit 1
fi

cfg_file=""
if [ $# == 1 ]; then
  cfg_file=$1
fi

export PYTHONPATH=.

python tools/eval.py -cfg "${cfg_file}" \
  --opt-level O1 \
  --evaluate
