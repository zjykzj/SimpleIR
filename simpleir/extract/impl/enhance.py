# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午5:22
@file: enhance.py
@author: zj
@description: 
"""
import os
import joblib

from enum import Enum
from numpy import ndarray
from sklearn.decomposition import PCA

import torch
from torch import Tensor

from simpleir.utils.misc import colorstr
from simpleir.utils.norm import l2_norm
from simpleir.utils.logger import LOGGER


class EnhanceType(Enum):
    IDENTITY = 'IDENTITY'
    L2_NORM = "L2_NORM"
    PCA = 'PCA'  # L2_Norm -> PCA -> L2_Norm
    PCA_W = 'PCA_W'  # L2_Norm -> PCA_Whiten -> L2_Norm


def pca_fit(feat_array: ndarray, rd: int = 512, is_whiten: bool = False) -> PCA:
    """
    Calculate pca/whitening parameters
    """
    # Normalize
    feat_array = l2_norm(torch.from_numpy(feat_array)).numpy()

    # Whiten and reduce dimension
    pca_model = PCA(n_components=rd, whiten=is_whiten)
    pca_model.fit(feat_array)

    return pca_model


def do_enhance(feat_tensor: Tensor,
               enhance_type: EnhanceType = EnhanceType.IDENTITY,
               is_gallery: bool = False,
               learn_pca: bool = True, pca_path: str = None, reduce_dimension: int = 512) -> torch.Tensor:
    """
    Feature enhancement
    """
    if enhance_type is EnhanceType.IDENTITY:
        return feat_tensor
    elif enhance_type is EnhanceType.L2_NORM:
        return l2_norm(feat_tensor)
    elif enhance_type is EnhanceType.PCA or enhance_type is EnhanceType.PCA_W:
        if is_gallery and learn_pca:
            is_whiten = enhance_type is EnhanceType.PCA_W

            pca_model = pca_fit(feat_tensor.numpy(), rd=reduce_dimension, is_whiten=is_whiten)
            LOGGER.info(f'Saving PCA model: {colorstr(pca_path)}')
            joblib.dump(pca_model, pca_path)
        else:
            assert os.path.isfile(pca_path), pca_path
            LOGGER.info(f'Loading PCA model: {colorstr(pca_path)}')
            pca_model = joblib.load(pca_path)

        # Normalize
        feat_tensor = l2_norm(feat_tensor)
        # PCA
        feat_array = pca_model.transform(feat_tensor.numpy())
        # Normalize
        feat_tensor = l2_norm(torch.from_numpy(feat_array))

        return feat_tensor
    else:
        raise ValueError(f'{enhance_type} does not support')
