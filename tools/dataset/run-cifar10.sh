#!/bin/bash

set -eux

folder="../datasets"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

echo "Download CIFAR10 and Extract"
python3 tools/dataset/extract_torchvision_dataset.py "${folder}/cifar10/" --dataset CIFAR10
