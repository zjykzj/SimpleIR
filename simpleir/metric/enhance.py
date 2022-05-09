# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午5:22
@file: enhance.py
@author: zj
@description: 
"""
import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch import norm


# def l2_norm(feats: np.ndarray) -> np.ndarray:
def l2_norm(feats: torch.Tensor) -> torch.Tensor:
    if len(feats.shape) == 1:
        # feats = np.array([feats])
        feats = feats.reshape(1, -1)

    # return normalize(feats, norm="l2", return_norm=False)
    return feats / torch.norm(feats, dim=1, keepdim=True)


# def enhance(feats: np.ndarray, enhance_type='normal') -> np.ndarray:
def enhance(feats: torch.Tensor, enhance_type='normal') -> torch.Tensor:
    """
    Calculate similarity (Euclidean distance / Cosine distance)
    """
    assert enhance_type in ['normal', 'l2-norm']

    if enhance_type == 'normal':
        return feats
    elif enhance_type == 'l2-norm':
        return l2_norm(feats)
    else:
        raise ValueError(f'{enhance_type} does not support')
