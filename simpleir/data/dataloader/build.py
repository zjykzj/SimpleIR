# -*- coding: utf-8 -*-

"""
@date: 2022/4/8 上午10:47
@file: buld.py
@author: zj
@description: 
"""

from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data._utils.collate import default_collate

import numpy as np
from yacs.config import CfgNode

from zcls2.data.dataloader.collate import fast_collate


def custom_fn(batches: List) -> Tuple[Tensor, Tensor, List]:
    images = [batch[0] for batch in batches]
    targets = [batch[1] for batch in batches]
    paths = [batch[2] for batch in batches]

    if isinstance(targets[0], int):
        targets = torch.from_numpy(np.array(targets))
        return torch.stack(images), targets, paths
    else:
        return torch.stack(images), torch.stack(targets), paths


def build_dataloader(cfg: CfgNode, dataset: Dataset,
                     sampler: Sampler = None, shuffle: bool = False,
                     is_train: bool = True, w_path: bool = False) -> DataLoader:
    batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE if is_train else cfg.DATALOADER.TEST_BATCH_SIZE
    num_workers = cfg.DATALOADER.NUM_WORKERS if is_train else 0

    if cfg.CHANNELS_LAST:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    if w_path:
        collate_fn = custom_fn
    else:
        if cfg.DATALOADER.COLLATE_FN == 'fast':
            collate_fn = lambda b: fast_collate(b, memory_format)
        else:
            collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, sampler=sampler, collate_fn=collate_fn)

    return data_loader
