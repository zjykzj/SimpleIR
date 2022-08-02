# -*- coding: utf-8 -*-

"""
@date: 2022/8/2 下午5:11
@file: build.py
@author: zj
@description: 
"""

from yacs.config import CfgNode
from torch.utils.data import Sampler, Dataset

from . import pk_sampler


def build_sampler(cfg: CfgNode, dataset: Dataset) -> Sampler:
    name = cfg.SAMPLER.NAME

    if name == '':
        return None

    if name in pk_sampler.__all__:
        # targets is a list where the i_th element corresponds to the label of i_th dataset element.
        # This is required for PKSampler to randomly sample from exactly p classes. You will need to
        # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
        # targets attribute with the same format.
        targets = dataset.targets
        p = cfg.SAMPLER.LABELS_PER_BATCH
        k = cfg.SAMPLER.SAMPLES_PER_LABEL

        sampler = pk_sampler.__dict__[name](targets, p, k)
    else:
        raise ValueError(f"{name} does not support")

    return sampler
