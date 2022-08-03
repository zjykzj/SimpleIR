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

from .impl import cccf, image_folder, general_dataset, oxford

__all__ = ['build_dataset']


def build_dataset(cfg: CfgNode,
                  transform: Optional[transforms.Compose] = None,
                  target_transform: Optional[transforms.Compose] = None,
                  is_train: Optional[bool] = True,
                  is_gallery: Optional[bool] = False,
                  w_path: Optional[bool] = False,
                  ) -> Dataset:
    if is_train:
        data_root = cfg.DATASET.TRAIN_ROOT
        dataset_name = cfg.DATASET.NAME
    elif is_gallery:
        data_root = cfg.DATASET.GALLERY_DIR
        dataset_name = cfg.DATASET.RETRIEVAL_NAME
    else:
        data_root = cfg.DATASET.QUERY_DIR
        dataset_name = cfg.DATASET.RETRIEVAL_NAME

    w_path = w_path or not is_train

    # Data loading code
    if dataset_name in image_folder.__all__:
        dataset = image_folder.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, w_path=w_path
        )
    elif dataset_name in cccf.__all__:
        dataset = cccf.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, train=is_train, w_path=w_path,
            is_gallery=is_gallery
        )
    elif dataset_name in general_dataset.__all__:
        dataset = general_dataset.__dict__[dataset_name](
            data_root, transform=transform, target_transform=target_transform, w_path=w_path
        )
    elif dataset_name in oxford.__all__:
        dataset = oxford.__dict__[dataset_name](
            data_root, is_gallery=is_gallery, transform=transform, target_transform=target_transform, w_path=w_path
        )
    else:
        raise ValueError(f"{dataset_name} do not support")

    return dataset
