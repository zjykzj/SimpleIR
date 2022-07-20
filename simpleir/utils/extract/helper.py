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

from torch.nn import Module
from torch.utils.data import DataLoader

from .extractor import Extractor
from .aggregator import Aggregator
from .enhancer import Enhancer


def save_features(feat_array: ndarray, feat_name_list: List, feature_dir: str) -> None:
    assert os.path.isdir(feature_dir), feature_dir

    for feat, feat_name in zip(feat_array, feat_name_list):
        feat_path = os.path.join(feature_dir, f'{feat_name}.npy')

        if os.path.isfile(feat_path):
            os.remove(feat_path)
        np.save(feat_path, feat)


class ExtractHelper(object):

    def __init__(self, model: Module = None, model_arch: str = 'resnet50', pretrained: str = None, layer: str = 'fc',
                 data_loader: DataLoader = None, save_dir: str = None,
                 aggregate_type: str = 'IDENTITY', enhance_type: str = 'IDENTITY', reduce_dimension: int = 512):
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

        self.extractor = Extractor(model, data_loader)
        self.aggregator = Aggregator(aggregate_type=self.aggregate_type)
        self.enhancer = Enhancer(enhance_type=self.enhance_type, reduce_dimension=self.reduce_dimension,
                                 save_dir=self.save_dir)

    def run(self):
        print("Extract features ...")
        image_name_list, target_list, feat_tensor = self.extractor.run()

        print("Aggregate features ...")
        aggregated_tensor = self.aggregator.run(feat_tensor).reshape(feat_tensor.shape[0], -1)

        print(f"Enhance features ...")
        enhanced_tensor = self.enhancer.run(aggregated_tensor)

        print("Save features ...")
        save_features(enhanced_tensor.numpy(), image_name_list, self.save_dir)

        content_dict = OrderedDict()
        for image_name, target in zip(image_name_list, target_list):
            content_dict[image_name] = target

        info_path = os.path.join(self.save_dir, 'info.pkl')
        print(f'Save to {info_path}')
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
