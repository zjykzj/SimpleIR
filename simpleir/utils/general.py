# -*- coding: utf-8 -*-

"""
@date: 2023/8/20 下午2:40
@file: general.py
@author: zj
@description: 
"""

import os
import pickle

import numpy as np
from pathlib import Path


def save_features(classes, image_name_list, label_list, feat_tensor_list, feat_dir, info_path):
    content_dict = dict()
    for image_name, target, feat_tensor in zip(image_name_list, label_list, feat_tensor_list):
        cls_name = classes[target]
        cls_dir = os.path.join(feat_dir, cls_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        feat_path = os.path.join(cls_dir, image_name + ".npy")
        np.save(feat_path, feat_tensor.numpy())

        assert image_name not in content_dict.keys()
        content_dict[image_name] = {
            'path': feat_path,
            'class': cls_name,
            'label': target
        }

    info_dict = {
        'content': content_dict,
        'classes': classes,
    }
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)


def load_features(info_path: str):
    """
    遍历所有目录，获得最后的文件夹

    query_feat_list, query_label_list, query_img_name_list, query_cls_list
    """
    assert os.path.isfile(info_path), info_path

    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    feat_list = list()
    label_list = list()
    img_name_list = list()
    for img_name, v_dict in info_dict['content'].items():
        feat_path = v_dict['path']
        feat = np.load(feat_path)
        feat_list.append(feat)

        label = v_dict['label']
        label_list.append(label)

        img_name_list.append(img_name)

    classes = info_dict['classes'] if 'classes' in info_dict.keys() else None
    return feat_list, label_list, img_name_list, classes


def save_retrieval(gallery, query, content_dict, query_classes, save_dir):
    info_dict = {
        'content': content_dict,
        'query_path': query,
        'gallery_path': gallery
    }
    if query_classes is not None:
        info_dict['classes'] = query_classes
    info_path = os.path.join(save_dir, 'info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)


def load_retrieval(retrieval_path: str):
    with open(retrieval_path, 'rb') as f:
        info_dict = pickle.load(f)

    batch_rank_name_list = list()
    batch_rank_label_list = list()
    query_name_list = list()
    query_label_list = list()
    for query_img_name, (query_label, rank_img_name_list, rank_label_list) in info_dict['content'].items():
        query_name_list.append(query_img_name)
        query_label_list.append(query_label)
        batch_rank_name_list.append(rank_img_name_list)
        batch_rank_label_list.append(rank_label_list)

    return batch_rank_name_list, batch_rank_label_list, query_name_list, query_label_list
