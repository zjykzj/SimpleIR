# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午5:22
@file: enhance.py
@author: zj
@description: 
"""
import numpy as np
from sklearn.preprocessing import normalize


def l2_norm(feats: np.ndarray) -> np.ndarray:
    if len(feats.shape) == 1:
        feats = np.array([feats])

    return normalize(feats, norm="l2", return_norm=False)


def enhance(feats: np.ndarray, enhance_type='normal') -> np.ndarray:
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
