# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:05
@file: caltech101.py
@author: zj
@description: 假定Caltech101的数据路径为
data_root/
    class1/
        img1
        img2
    class2/
        img1
        img2
    ...

划分每类数据为查询集和图库，
1. 如果该类数据数目大于等于10张，则按2：8比例划分查询集和图库
2. 如果该类数据数目小于10张，则按1:9比例划分查询集和图库
注意：每类数据至少拥有一张查询集图像
"""

import os
import glob
import random
import shutil
import time

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from simpleir.configs.key_words import KEY_GALLERY, KEY_QUERY

__all__ = ['Caltech101']


class Caltech101:

    def __init__(self, ):
        pass

    def _save_to_dst(self, img_list, class_name, dst_root):
        dst_class_dir = os.path.join(dst_root, class_name)
        if not os.path.exists(dst_class_dir):
            os.makedirs(dst_class_dir)

        for img_path in img_list:
            assert os.path.isfile(img_path), img_path
            img_name = os.path.split(img_path)[1]
            dst_img_path = os.path.join(dst_class_dir, img_name)

            shutil.copy(img_path, dst_img_path)

    def run(self, src_root, dst_root):
        assert os.path.isdir(src_root), src_root
        classes = os.listdir(src_root)

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        dst_gallery = os.path.join(dst_root, KEY_GALLERY)
        dst_query = os.path.join(dst_root, KEY_QUERY)

        start = time.time()
        for class_name in tqdm(classes):
            cls_dir = os.path.join(src_root, class_name)
            assert os.path.isdir(cls_dir), cls_dir

            img_list = glob.glob(os.path.join(cls_dir, '*.jpg'))
            if len(img_list) <= 1:
                self._save_to_dst(img_list, class_name, dst_gallery)
            elif len(img_list) <= 10:
                query_img_path = random.choice(img_list)
                query_list = [query_img_path]
                gallery_list = list(set(img_list) - set(query_list))

                self._save_to_dst(gallery_list, class_name, dst_gallery)
                self._save_to_dst(query_list, class_name, dst_query)
            else:
                gallery_list, query_list = train_test_split(img_list, test_size=0.2, train_size=0.8)
                self._save_to_dst(gallery_list, class_name, dst_gallery)
                self._save_to_dst(query_list, class_name, dst_query)
        end = time.time()
        print('time:', (end - start))
