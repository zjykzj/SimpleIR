# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午9:42
@file: new_extractor.py
@author: zj
@description: 
"""
from typing import List

import os
import pickle

import numpy as np
from numpy import ndarray

from collections import OrderedDict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from simpleir.utils.extract.impl.extractor import Extractor
from simpleir.utils.extract.impl.aggregator import Aggregator
from simpleir.utils.extract.impl.enhancer import Enhancer

from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['ExtractHelper']


def save_features(feat_array: ndarray, feat_name_list: List, feature_dir: str) -> None:
    assert os.path.isdir(feature_dir), feature_dir

    for feat, feat_name in zip(feat_array, feat_name_list):
        feat_path = os.path.join(feature_dir, f'{feat_name}.npy')

        if os.path.isfile(feat_path):
            os.remove(feat_path)
        np.save(feat_path, feat)


class ExtractHelper(object):

    def __init__(self, model: Module = None, device=torch.device('cpu'),
                 model_arch: str = 'resnet50', pretrained: str = None, layer: str = 'fc',
                 data_loader: DataLoader = None, save_dir: str = None, is_gallery=False,
                 aggregate_type: str = 'IDENTITY', enhance_type: str = 'IDENTITY',
                 learn_pca: bool = True, pca_path: str = None, reduce_dimension: int = 512):
        assert model is not None
        assert data_loader is not None
        assert os.path.exists(save_dir), save_dir

        self.model_arch = model_arch
        self.pretrained = pretrained
        self.layer = layer

        self.classes = data_loader.dataset.classes

        self.save_dir = save_dir
        self.aggregate_type = aggregate_type
        self.enhance_type = enhance_type
        self.reduce_dimension = reduce_dimension

        self.extractor = Extractor(model, data_loader, device)
        self.aggregator = Aggregator(aggregate_type=self.aggregate_type)
        self.enhancer = Enhancer(
            enhance_type=self.enhance_type, is_gallery=is_gallery, save_dir=self.save_dir,
            learn_pca=learn_pca, pca_path=pca_path, reduce_dimension=self.reduce_dimension,
        )

    def run(self):
        image_name_list, target_list, feat_tensor = self.extractor.run()

        aggregated_tensor = self.aggregator.run(feat_tensor).reshape(feat_tensor.shape[0], -1)

        enhanced_tensor = self.enhancer.run(aggregated_tensor)

        save_features(enhanced_tensor.numpy(), image_name_list, self.save_dir)

        content_dict = OrderedDict()
        for image_name, target in zip(image_name_list, target_list):
            content_dict[image_name] = target

        info_path = os.path.join(self.save_dir, 'info.pkl')
        logger.info(f'Save to {info_path}')
        info_dict = {
            'model': self.model_arch,
            'pretrained': self.pretrained,
            'classes': self.classes,
            'feat': self.layer,
            'aggregate': self.aggregate_type,
            'enhance': self.enhance_type,
            'content': content_dict
        }
        with open(info_path, 'wb') as f:
            pickle.dump(info_dict, f)
