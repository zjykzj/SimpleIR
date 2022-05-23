# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午11:00
@file: helper.py
@author: zj
@description: 图像特征提取辅助类
1. 批量计算图像特征
2. 批量保存图像特征

每条图像路径对应图像特征
"""

import os
import time
import pickle

from tqdm import tqdm
from simpleir.configs.key_words import KEY_FEAT


def save_part_feat(feat_dict, part_file_path):
    assert not os.path.isfile(part_file_path), part_file_path

    file_dir = os.path.split(part_file_path)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(part_file_path, 'wb') as f:
        pickle.dump(feat_dict, f)


class ExtractHelper:
    """
    A helper class to extract feature maps from model, and then aggregate them.
    """

    def __init__(self, data_loader, model, feature):
        self.data_loader = data_loader
        self.model = model
        self.feature = feature

    def run(self, dst_root, save_prefix='', save_interval: int = 5000):
        if save_prefix != '':
            save_prefix = save_prefix + '_'

        feat_dict = dict()
        feat_dict['classes'] = self.data_loader.dataset.classes
        feat_dict['feats'] = list()
        feat_num = 0

        part_count = 0
        start = time.time()
        for batch in tqdm(self.data_loader):
            images, targets, paths = batch

            # 提取特征
            outputs = self.model(images)[KEY_FEAT].detach().cpu()
            feats = self.feature.run(outputs)

            for path, target, feat in zip(paths, targets, feats):
                feat_dict['feats'].append({
                    'path': path,
                    'label': target,
                    'feat': feat
                })
                feat_num += 1
            if feat_num > save_interval:
                save_part_feat(feat_dict, os.path.join(dst_root, f'{save_prefix}part_{part_count}.csv'))
                part_count += 1

                del feat_dict
                feat_dict = dict()
                feat_dict['classes'] = self.data_loader.dataset.classes
                feat_dict['feats'] = list()
                feat_num = 0
        if feat_num > 1:
            save_part_feat(feat_dict, os.path.join(dst_root, f'{save_prefix}part_{part_count}.csv'))
        end = time.time()
        print('time: ', end - start)
