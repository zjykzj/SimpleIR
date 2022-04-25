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

from zcls2.data.transform.build import build_transform
from zcls2.data.dataset.build import build_dataset
from zcls2.data.sampler.build import build_sampler

from .dataloader.build import build_dataloader


def build_data(cfg: CfgNode) -> Tuple[Sampler, DataLoader, DataLoader]:
    train_transform, train_target_transform = build_transform(cfg, is_train=True)
    train_dataset = build_dataset(cfg, train_transform, train_target_transform, is_train=True)

    val_transform, val_target_transform = build_transform(cfg, is_train=False)
    val_dataset = build_dataset(cfg, val_transform, val_target_transform, is_train=False)

    if isinstance(train_dataset, IterableDataset):
        train_sampler, val_sampler = None, None
        shuffle = False
    else:
        train_sampler, val_sampler = build_sampler(cfg, train_dataset, val_dataset)
        shuffle = train_sampler is None

    train_loader, val_loader = build_dataloader(cfg,
                                                train_dataset, val_dataset, train_sampler, val_sampler,
                                                shuffle)

    return train_sampler, train_loader, val_loader
