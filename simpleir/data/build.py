# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: data.py
@author: zj
@description: 
"""

from typing import Tuple

from yacs.config import CfgNode
from torch.utils.data import IterableDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from zcls2.data.transform.build import build_transform

from .dataloader.build import build_dataloader
from .dataset.build import build_dataset


def build_data(cfg: CfgNode, w_path: bool = False) -> Tuple[Sampler, DataLoader, DataLoader]:
    train_transform, train_target_transform = build_transform(cfg, is_train=True)
    train_dataset = build_dataset(cfg, train_transform, train_target_transform, is_train=True, w_path=w_path)

    val_transform, val_target_transform = build_transform(cfg, is_train=False)
    val_dataset = build_dataset(cfg, val_transform, val_target_transform, is_train=False)

    train_sampler = None
    val_sampler = None
    if isinstance(train_dataset, IterableDataset):
        shuffle = False
    else:
        if cfg.DISTRIBUTED:
            train_sampler = DistributedSampler(train_dataset)
        shuffle = train_sampler is None

    train_loader, val_loader = build_dataloader(cfg,
                                                train_dataset, val_dataset, train_sampler, val_sampler,
                                                shuffle)

    return train_sampler, train_loader, val_loader
