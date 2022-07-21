# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午11:00
@file: Retriever.py
@author: zj
@description:
"""
from typing import Tuple, List

import os

from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from simpleir.configs.key_words import KEY_FEAT
from zcls2.util import logging

logger = logging.get_logger(__name__)


class Extractor:

    def __init__(self, model: Module, data_loader: DataLoader, device=torch.device('cpu')):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def run(self) -> Tuple[List, list, Tensor]:
        image_name_list = list()
        target_list = list()
        feat_list = list()
        for images, targets, paths in tqdm(self.data_loader):
            res_dict = self.model.forward(images.to(self.device))
            feats_tensor = res_dict[KEY_FEAT].detach().cpu()

            for path, target, feat in zip(paths, targets, feats_tensor):
                image_name = os.path.splitext(os.path.split(path)[1])[0]

                image_name_list.append(image_name)
                target_list.append(target)
                feat_list.append(feat)

        return image_name_list, target_list, torch.stack(feat_list)
