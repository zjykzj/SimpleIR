#!/bin/bash

set -eux

cd ../../

folder="./data"
if [ ! -d ${folder} ]; then
  mkdir ${folder}
fi

echo "Download MNIST and Extract"
python3 demo/dataset/extract_torchvision_dataset.py "${folder}/mnist/" --dataset MNIST
