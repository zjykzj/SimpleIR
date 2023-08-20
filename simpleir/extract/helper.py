# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午9:42
@file: new_extractor.py
@author: zj
@description: 
"""
from typing import List

import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from simpleir.extract.impl.extractor import Extractor
from simpleir.extract.impl.aggregator import Aggregator
from simpleir.extract.impl.enhancer import Enhancer

from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['ExtractHelper']


class ExtractHelper(object):
    """
    分三步进行，特征提取、特征集成、特征强化

    特征提取，输入模型，注册hook。在前向运行完成后提取特征
    特征集成，提取了多层卷积激活，然后进行集成操作
    """

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

        if hasattr(data_loader.dataset, "classes"):
            self.classes = data_loader.dataset.classes
        else:
            self.classes = None

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
