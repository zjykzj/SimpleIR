# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: data.py
@author: zj
@description: 
"""

from typing import Tuple, Optional, Any

from yacs.config import CfgNode
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from zcls2.data.transform.build import build_transform

from .dataloader.build import build_dataloader
from .dataset.build import build_dataset

__all__ = ['build_data']


def build_data(cfg: CfgNode, is_train: bool = True, is_gallery: bool = False, w_path: bool = False) -> Tuple[
    Optional[DistributedSampler[Any]], DataLoader]:
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform, target_transform, is_train=is_train, is_gallery=is_gallery, w_path=w_path)

    # By default, for train, shuffle the indexes; For test, output in sequence
    # If sampler is set, shuffle is not called
    sampler = None
    if not isinstance(dataset, IterableDataset) and cfg.DISTRIBUTED:
        sampler = DistributedSampler(dataset)
    shuffle = is_train and sampler is None

    # Ensure the consistency of output sequence. Set shuffle=False and num_workers=0 in test
    data_loader = build_dataloader(cfg, dataset, sampler, shuffle, is_train=is_train, w_path=w_path)

    return sampler, data_loader
