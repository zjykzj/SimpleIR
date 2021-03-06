# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午5:22
@file: enhance.py
@author: zj
@description: 
"""
import logging
import os
import joblib
import torch

from enum import Enum
from numpy import ndarray
from sklearn.decomposition import PCA

from simpleir.utils.norm import l2_norm

from zcls2.util import logging

logger = logging.get_logger(__name__)


class EnhanceType(Enum):
    IDENTITY = 'IDENTITY'
    L2_NORM = "L2_NORM"
    PCA = 'PCA'  # L2_Norm -> PCA -> L2_Norm
    PCA_W = 'PCA_W'  # L2_Norm -> PCA_Whiten -> L2_Norm


def pca_fit(feat_array: ndarray, rd=512, is_whiten=False) -> PCA:
    """
    Calculate pca/whitening parameters
    """
    # Normalize
    feat_array = l2_norm(torch.from_numpy(feat_array))

    # Whiten and reduce dimension
    pca_model = PCA(n_components=rd, whiten=is_whiten)
    pca_model.fit(feat_array)

    return pca_model


def do_enhance(feat_tensor: torch.Tensor, enhance_type: EnhanceType = EnhanceType.IDENTITY,
               reduce_dimension=512, is_gallery=False, save_dir=None, pca_path=None) -> torch.Tensor:
    """
    Feature enhancement
    """
    if enhance_type is EnhanceType.IDENTITY:
        return feat_tensor
    elif enhance_type is EnhanceType.L2_NORM:
        return l2_norm(feat_tensor)
    elif enhance_type is EnhanceType.PCA or enhance_type is EnhanceType.PCA_W:
        assert os.path.isdir(save_dir), save_dir
        assert pca_path is not None
        is_whiten = enhance_type is EnhanceType.PCA_W

        if is_gallery:
            pca_model = pca_fit(feat_tensor.numpy(), rd=reduce_dimension, is_whiten=is_whiten)
            logger.info('Saving PCA model: %s' % pca_path)
            joblib.dump(pca_model, pca_path)
        else:
            assert os.path.isfile(pca_path), pca_path
            logger.info('Loading PCA model: %s' % pca_path)
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


class Enhancer:

    def __init__(self, enhance_type='IDENTITY', reduce_dimension=512, is_gallery=False, save_dir=None, pca_path=None):
        self.enhance_type = EnhanceType[enhance_type]
        self.reduce_dimension = reduce_dimension
        self.is_gallery = is_gallery
        self.save_dir = save_dir
        self.pca_path = pca_path

    def run(self, feat_tensor: torch.Tensor):
        return do_enhance(feat_tensor, self.enhance_type,
                          reduce_dimension=self.reduce_dimension,
                          is_gallery=self.is_gallery,
                          save_dir=self.save_dir,
                          pca_path=self.pca_path)
