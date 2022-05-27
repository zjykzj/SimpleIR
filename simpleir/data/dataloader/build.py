# -*- coding: utf-8 -*-

"""
@date: 2022/4/8 上午10:47
@file: buld.py
@author: zj
@description: 
"""
from typing import Tuple, Any

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

from yacs.config import CfgNode

from zcls2.data.dataloader.collate import fast_collate


def build_dataloader(cfg: CfgNode, train_dataset: Dataset, val_dataset: Dataset,
                     train_sampler: Sampler = None, val_sampler: Sampler = None,
                     shuffle: bool = False) -> Tuple[Any, Any]:
    train_batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE
    test_batch_size = cfg.DATALOADER.TEST_BATCH_SIZE
    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.CHANNELS_LAST:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    if cfg.DATALOADER.COLLATE_FN == 'fast':
        collate_fn = lambda b: fast_collate(b, memory_format)
    else:
        collate_fn = default_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    # Ensure the consistency of output sequence. Set shuffle=False and num_workers=0
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    return train_loader, val_loader
