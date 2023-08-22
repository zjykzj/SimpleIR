# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午9:42
@file: new_extractor.py
@author: zj
@description: 
"""
from typing import List, Tuple, Dict

import os
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from simpleir.utils.logger import LOGGER
from .impl.aggregate import AggregateType, do_aggregate
from .impl.enhance import EnhanceType, do_enhance

__all__ = ['ExtractHelper']


class ExtractHelper(object):
    """
    分三步进行，特征提取、特征集成、特征强化

    特征提取，输入模型，注册hook。在前向运行完成后提取特征
    特征集成，提取卷积激活进行集成操作
    特征强化，
    第一种情况：针对gallery特征进行学习，然后进行维度缩减
    第二种情况：针对query特征进行使用，然后进行维度缩减
    第三种情况：外部指定一个pca，这种情况下就不需要学习了，直接进行操作即可

    需要保存吗？
    存放到指定位置即可。怎么简单怎么来，先把整个流程打通。

    对于图像检索，需要什么？每个类对应的id

    特征提取阶段可以提供什么？
    1. 对于检索库
        1. 每个检索特征的保存地址
        2. 每个检索特征对应的类别名/类别标签
    2. 对于查询库
        1. 每个查询特征的保存地址
        2. 每个查询特征对应的类别名/类别标签
    """

    def __init__(self,
                 # 特征提取
                 model: Module,
                 target_layer: Module,
                 device: torch.device,
                 # 特征集成
                 aggregate_type: str = 'IDENTITY',
                 # 特征增强
                 enhance_type: str = 'IDENTITY',
                 learn_pca: bool = True,
                 reduce_dimension: int = 512,
                 pca_path: str = None,
                 ):
        # Extract
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        # Aggregate
        self.aggregate_type = AggregateType[aggregate_type]
        # Enhance
        self.enhance_type = EnhanceType[enhance_type]
        self.learn_pca = learn_pca
        self.pca_path = pca_path
        self.reduce_dimension = reduce_dimension

    def save_activation(self, module, input, output):
        activation = output
        self.activations = activation.cpu().detach()

    def run(self, dataloader: DataLoader, is_gallery: bool = True):
        image_name_list = list()
        label_list = list()
        feat_tensor_list = list()
        for images, targets, paths in tqdm(dataloader):
            _ = self.model.forward(images.to(self.device))

            activations = do_aggregate(self.activations, self.aggregate_type).reshape(len(images), -1)

            for path, target, feat_tensor in zip(paths, targets.numpy(), activations):
                image_name = os.path.splitext(os.path.split(path)[1])[0]

                image_name_list.append(image_name)
                label_list.append(int(target))
                feat_tensor_list.append(feat_tensor)

        feat_tensor_list = do_enhance(torch.stack(feat_tensor_list),
                                      self.enhance_type,
                                      learn_pca=self.learn_pca,
                                      reduce_dimension=self.reduce_dimension,
                                      pca_path=self.pca_path,
                                      is_gallery=is_gallery,
                                      )

        return image_name_list, label_list, feat_tensor_list
