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

from . import cccf, image_folder, general_dataset

__all__ = ['build_dataset']


def build_dataset(cfg: CfgNode,
                  transform: Optional[transforms.Compose] = None,
                  target_transform: Optional[transforms.Compose] = None,
                  is_train: Optional[bool] = True,
                  w_path: Optional[bool] = False,
                  ) -> Dataset:
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT

    w_path = w_path or not is_train

    # Data loading code
    if dataset_name in image_folder.__all__:
        dataset = image_folder.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, w_path=w_path
        )
    elif dataset_name in cccf.__all__:
        dataset = cccf.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, train=is_train, w_path=w_path
        )
    elif dataset_name in general_dataset.__all__:
        dataset = general_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, w_path=w_path
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
