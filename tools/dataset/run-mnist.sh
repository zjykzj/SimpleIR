#!/bin/bash

set -eux

folder="../datasets"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

echo "Download MNIST and Extract"
python3 tools/dataset/extract_torchvision_dataset.py "${folder}/mnist/" --dataset MNIST
