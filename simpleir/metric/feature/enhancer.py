# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午5:22
@file: enhance.py
@author: zj
@description: 
"""

import torch

from enum import Enum
from simpleir.utils.norm import l2_norm


class EnhanceType(Enum):
    IDENTITY = 'IDENTITY'
    L2_NORM = "L2_NORM"


def do_enhance(feats: torch.Tensor, enhance_type: EnhanceType = EnhanceType.IDENTITY) -> torch.Tensor:
    """
    Feature enhancement
    """
    if enhance_type is EnhanceType.IDENTITY:
        return feats
    elif enhance_type is EnhanceType.L2_NORM:
        return l2_norm(feats)
    else:
        raise ValueError(f'{enhance_type} does not support')
