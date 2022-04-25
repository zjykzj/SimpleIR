#!/bin/bash

set -eux

cd ../../

folder="./data"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

echo "Download CIFAR10 and Extract"
python3 demo/dataset/extract_torchvision_dataset.py "${folder}/cifar10/" --dataset CIFAR10
