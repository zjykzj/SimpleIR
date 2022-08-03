# -*- coding: utf-8 -*-

"""
@date: 2022/8/3 上午10:18
@file: distributed_pk_sampler.py
@author: zj
@description: See https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
"""
from typing import Iterator, Sized, Optional, List, Union, Dict

import random

import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Sampler
from torchvision.datasets.samplers import DistributedSampler

__all__ = ['DistributedPKSampler']


def create_groups(groups: List[int], k: int) -> Dict:
    """Bins sample indices with respect to groups, remove bins with less than k samples

    Args:
        groups (list[int]): where ith index stores ith sample's group id

    Returns:
        defaultdict[list]: Bins of sample indices, binned by group_idx
    """
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    keys_to_remove = []
    for key in group_samples:
        if len(group_samples[key]) < k:
            keys_to_remove.append(key)
            continue

    for key in keys_to_remove:
        group_samples.pop(key)

    return group_samples


class DistributedPKSampler(DistributedSampler):

    def __init__(self, dataset: Sized, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = False, group_size: int = 1, targets: List[int] = None, p: int = 8, k: int = 8) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, group_size)

        assert targets is not None
        self.targets = targets
        self.p = p
        self.k = k

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices: Union[torch.Tensor, List[int]]
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        total_group_size = self.total_size // self.group_size
        indices = torch.reshape(
            torch.LongTensor(indices), (total_group_size, self.group_size)
        )

        # subsample
        indices = indices[self.rank:total_group_size:self.num_replicas, :]
        indices = torch.reshape(indices, (-1,)).tolist()
        assert len(indices) == self.num_samples

        if isinstance(self.dataset, Sampler):
            orig_indices = list(iter(self.dataset))
            indices = [orig_indices[i] for i in indices]

        # Group sampling
        groups = np.array(self.targets)[indices].tolist()
        self.groups = create_groups(groups, self.k)

        # Ensures there are enough classes to sample from
        if len(self.groups) < self.p:
            raise ValueError("There are not enough classes to sample from")

        # Randomly disrupt the arrangement of images in the number of each category
        # Shuffle samples within groups
        for key in self.groups:
            random.shuffle(self.groups[key])

        # Keep track of the number of samples left for each group
        group_samples_remaining = {}
        for key in self.groups:
            group_samples_remaining[key] = len(self.groups[key])

        indices = list()
        indices_remaining = list()
        while len(group_samples_remaining) > self.p:
            # Select p groups at random from valid/remaining groups
            group_ids = list(group_samples_remaining.keys())
            # Randomly sample P categories from the number of all categories
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            for i in selected_group_idxs:
                group_id = group_ids[i]
                group = self.groups[group_id]
                # Get the first k images of a category
                for _ in range(self.k):
                    # No need to pick samples at random since group samples are shuffled
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    # yield group[sample_idx]
                    indices.append(group[sample_idx])
                    group_samples_remaining[group_id] -= 1

                # Don't sample from group if it has less than k samples remaining
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)

                    # Get the remaining images
                    for _ in range(group_samples_remaining[group_id]):
                        # No need to pick samples at random since group samples are shuffled
                        sample_idx = len(group) - group_samples_remaining[group_id]
                        indices_remaining.append(group[sample_idx])
                        group_samples_remaining[group_id] -= 1

        random.shuffle(indices_remaining)
        indices.extend(indices_remaining)

        return iter(indices)

    def __len__(self) -> int:
        return super().__len__()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
