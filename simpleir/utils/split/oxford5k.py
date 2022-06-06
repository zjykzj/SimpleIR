# -*- coding: utf-8 -*-

"""
@date: 2022/6/6 下午2:35
@file: oxford.py
@author: zj
@description: 
"""

import os
import time
import shutil

from simpleir.configs.key_words import KEY_GALLERY, KEY_QUERY

__all__ = ['Oxford5k']


class Oxford5k:
    split_file = 'tools/eval/split_file/oxford_split.txt'

    def __init__(self, ):
        assert os.path.isfile(self.split_file)

    def _save_to_dst(self, img_list, src_root, dst_root):
        assert os.path.isdir(src_root), src_root
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

        for img_path in img_list:
            src_img_path = os.path.join(src_root, img_path)
            assert os.path.isfile(src_img_path), img_path

            img_name = os.path.split(src_img_path)[1]
            dst_img_path = os.path.join(dst_root, img_name)

            shutil.copy(src_img_path, dst_img_path)

    def parse_txt(self):
        query_list = list()
        gallery_list = list()
        with open(self.split_file, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue

                img_path, flag = line.strip().split(' ')
                if int(flag) == 0:
                    query_list.append(img_path)
                elif int(flag) == 1:
                    gallery_list.append(img_path)
                else:
                    raise ValueError(f'ERROR: {line}')

        return query_list, gallery_list

    def run(self, src_root, dst_root):
        assert os.path.isdir(src_root), src_root

        if not os.path.exists(dst_root):
            os.makedirs(dst_root)
        dst_gallery = os.path.join(dst_root, KEY_GALLERY)
        dst_query = os.path.join(dst_root, KEY_QUERY)

        start = time.time()
        query_list, gallery_list = self.parse_txt()
        self._save_to_dst(query_list, src_root, dst_query)
        self._save_to_dst(gallery_list, src_root, dst_gallery)
        end = time.time()
        print('time:', (end - start))
