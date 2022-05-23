# -*- coding: utf-8 -*-

"""
@date: 2022/5/23 下午7:10
@file: general.py
@author: zj
@description: 
"""

import os
import shutil
import time
import random

from zcls2.config.key_word import KEY_DATASET, KEY_CLASSES, KEY_SEP

__all__ = ['GeneralSplitter']


def load_csv(csv_path):
    assert os.path.isfile(csv_path), csv_path

    data_dict = dict()
    with open(csv_path, 'r') as f:
        for line in f:
            img_path, label = line.strip().split(KEY_SEP)
            if label not in data_dict.keys():
                data_dict[label] = list()

            data_dict[label].append(img_path)

    return data_dict


def save_csv(data_dict, csv_path):
    assert not os.path.isfile(csv_path), csv_path

    data_list = list()
    for label, img_path_list in data_dict.items():
        data_list.extend([[label, img_path] for img_path in img_path_list])

    data_list_num = len(data_list)
    with open(csv_path, 'w') as f:
        for idx, (label, img_path) in enumerate(data_list):
            if idx < (data_list_num - 1):
                f.write(f"{img_path}{KEY_SEP}{label}\n")
            else:
                f.write(f"{img_path}{KEY_SEP}{label}")


class GeneralSplitter:

    def __init__(self, max_num=20) -> None:
        super().__init__()

        self.max_num = max_num

    def run(self, src_root, dst_root, ):
        assert os.path.isdir(src_root), src_root
        start = time.time()

        src_data_path = os.path.join(src_root, KEY_DATASET)
        src_cls_path = os.path.join(src_root, KEY_CLASSES)

        # read data.csv
        src_data_dict = load_csv(src_data_path)

        # random choice
        dst_data_dict = dict()
        random.seed(0)
        for label, img_path_list in src_data_dict.items():
            if len(img_path_list) <= self.max_num:
                dst_data_dict[label] = img_path_list
            else:
                dst_data_dict[label] = random.choices(img_path_list, k=self.max_num)

        # save
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        dst_data_path = os.path.join(dst_root, KEY_DATASET)
        dst_cls_path = os.path.join(dst_root, KEY_CLASSES)

        save_csv(dst_data_dict, dst_data_path)
        shutil.copy(src_cls_path, dst_cls_path)

        end = time.time()
        print('process time:', (end - start))
