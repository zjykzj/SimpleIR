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

from tqdm import tqdm
import numpy as np
from zcls.config.key_word import KEY_SEP


def save_part_feat(save_feat_list, part_file_path):
    assert not os.path.isfile(part_file_path), part_file_path

    file_dir = os.path.split(part_file_path)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    length = len(save_feat_list)
    with open(part_file_path, 'w') as f:
        for idx, item in enumerate(save_feat_list):
            path, output = item[:2]

            output_str = f'{KEY_SEP}'.join(np.array(output, dtype=str))
            if idx < (length - 1):
                f.write(f'{path}{KEY_SEP}{output_str}\n')
            else:
                f.write(f'{path}{KEY_SEP}{output_str}')


class ExtractHelper:
    """
    A helper class to extract feature maps from model, and then aggregate them.
    """

    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model

    def run(self, dst_root, save_interval: int = 5000):
        save_feat_list = list()
        part_count = 0

        start = time.time()
        for batch in tqdm(self.data_loader):
            images, targets, paths = batch

            # 提取特征
            outputs = self.model(images).detach().cpu().numpy()

            for path, output in zip(paths, outputs):
                save_feat_list.append([path, output])
            if len(save_feat_list) > save_interval:
                save_part_feat(save_feat_list, os.path.join(dst_root, f'part_{part_count}.csv'))
                part_count += 1
                del save_feat_list
                save_feat_list = list()
        if len(save_feat_list) > 1:
            save_part_feat(save_feat_list, os.path.join(dst_root, f'part_{part_count}.csv'))
        end = time.time()
        print('time: ', end - start)
