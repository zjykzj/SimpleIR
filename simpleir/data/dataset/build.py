# -*- coding: utf-8 -*-

"""
@date: 2022/5/20 下午3:09
@file: build.py
@author: zj
@description: 
"""

from typing import Optional
from yacs.config import CfgNode

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from . import cccf, image_folder

__all__ = ['build_dataset']


def build_dataset(cfg: CfgNode,
                  transform: Optional[transforms.Compose] = None,
                  target_transform: Optional[transforms.Compose] = None,
                  is_train: Optional[bool] = True) -> Dataset:
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT

    # Data loading code
    if dataset_name in image_folder.__all__:
        dataset = image_folder.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, is_path=not is_train
        )
    elif dataset_name in cccf.__all__:
        dataset = cccf.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, train=is_train, is_path=not is_train
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
