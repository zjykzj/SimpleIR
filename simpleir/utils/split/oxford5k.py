# -*- coding: utf-8 -*-

"""
@date: 2022/6/6 下午2:35
@file: oxford.py
@author: zj
@description: 
"""

import os
import time

import numpy as np

from simpleir.configs.key_words import KEY_GALLERY, KEY_QUERY
from zcls2.config.key_word import KEY_DATASET, KEY_CLASSES, KEY_SEP

__all__ = ['Oxford5k']


def parse_class(img_path: str) -> str:
    assert img_path.endswith('.jpg'), img_path

    img_name = os.path.split(img_path)[1]
    img_name_wo_suffix = os.path.splitext(img_name)[0]

    class_name = img_name_wo_suffix[:-7]
    return class_name


class Oxford5k:
    split_file = 'tools/eval/split_file/oxford_split.txt'

    def __init__(self, ):
        assert os.path.isfile(self.split_file)

        self.query_list, self.gallery_list, self.class_list = self.parse_txt()

    def parse_txt(self):
        class_list = list()

        query_list = list()
        gallery_list = list()
        with open(self.split_file, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                img_path, flag = line.strip().split(' ')
                cls_name = parse_class(img_path)
                if cls_name not in class_list:
                    class_list.append(cls_name)

                if int(flag) == 0:
                    query_list.append(img_path)
                elif int(flag) == 1:
                    gallery_list.append(img_path)
                else:
                    raise ValueError(f'ERROR: {line}')

        return query_list, gallery_list, list(np.sort(class_list))

    def _save_to_dst(self, img_list, class_list, src_root, dst_root):
        assert os.path.isdir(src_root), src_root
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        dst_data_csv = os.path.join(dst_root, KEY_DATASET)
        dst_cls_csv = os.path.join(dst_root, KEY_CLASSES)

        img_list_len = len(img_list)
        with open(dst_data_csv, 'w') as f:
            for idx, img_path in enumerate(img_list):
                cls_name = parse_class(img_path)
                label = class_list.index(cls_name)

                src_img_path = os.path.join(src_root, img_path)
                assert os.path.isfile(src_img_path), img_path

                f_str = f'{src_img_path}{KEY_SEP}{label}'
                if idx < (img_list_len - 1):
                    f_str += '\n'
                f.write(f_str)

        np.savetxt(dst_cls_csv, np.array(class_list), fmt='%s', delimiter='')

    def run(self, src_root, dst_root):
        assert os.path.isdir(src_root), src_root

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        dst_gallery = os.path.join(dst_root, KEY_GALLERY)
        dst_query = os.path.join(dst_root, KEY_QUERY)

        start = time.time()
        self._save_to_dst(self.query_list, self.class_list, src_root, dst_query)
        self._save_to_dst(self.gallery_list, self.class_list, src_root, dst_gallery)
        end = time.time()
        print('time:', (end - start))
