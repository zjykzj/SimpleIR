# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: Retriever.py
@author: zj
@description: 
"""
from typing import Tuple, List

import os
import pickle

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch

from .impl.distancer import Distancer
from .impl.ranker import Ranker
from .impl.reranker import ReRanker

from zcls2.config.key_word import KEY_SEP
from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['RetrievalHelper']


def load_features(feat_dir: str) -> Tuple[List[List], List[int], List[str], List[str]]:
    assert os.path.isdir(feat_dir), feat_dir

    info_path = os.path.join(feat_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    feat_list = list()
    label_list = list()
    img_name_list = list()
    for img_name, label in tqdm(info_dict['content'].items()):
        feat_path = os.path.join(feat_dir, f'{img_name}.npy')
        feat = np.load(feat_path)

        feat_list.append(list(feat))
        label_list.append(label)
        img_name_list.append(img_name)

    return feat_list, label_list, img_name_list, list(info_dict['classes'])


class RetrievalHelper:

    def __init__(self, query_dir: str, gallery_dir: str, save_dir: str, top_k=None,
                 distance_type: str = 'EUCLIDEAN', rank_type: str = 'NORMAL', re_rank_type='IDENTITY',
                 ):
        self.query_dir = query_dir
        assert os.path.isdir(self.query_dir), self.query_dir
        self.gallery_dir = gallery_dir
        assert os.path.isdir(self.gallery_dir), self.gallery_dir

        self.save_dir = save_dir
        assert os.path.isdir(self.save_dir), self.save_dir
        self.top_k = top_k

        self.distancer = Distancer(distance_type)
        self.ranker = Ranker(rank_type, top_k=self.top_k)
        self.reranker = ReRanker(re_rank_type)

    def run(self):
        logger.info(f"Loading query features from {self.query_dir}")
        query_feat_list, query_label_list, query_img_name_list, query_cls_list = load_features(self.query_dir)
        logger.info(f"Loading gallery features from {self.gallery_dir}")
        gallery_feat_list, gallery_label_list, gallery_img_name_list, gallery_cls_list = load_features(self.gallery_dir)
        assert query_cls_list == gallery_cls_list

        gallery_feat_tensor = torch.from_numpy(np.array(gallery_feat_list))
        gallery_target_tensor = torch.from_numpy(np.array(gallery_label_list))

        logger.info('Retrieval ...')
        content_dict = OrderedDict()
        assert self.top_k is None or (0 < self.top_k <= len(query_feat_list))

        for query_feat, query_label, query_img_name in tqdm(
                zip(query_feat_list, query_label_list, query_img_name_list), total=len(query_feat_list)):
            query_feat_tensor = torch.from_numpy(np.array(query_feat)).unsqueeze(0)

            batch_dists_tensor = self.distancer.run(query_feat_tensor, gallery_feat_tensor)

            batch_sort_idx_list, batch_rank_label_list = \
                self.ranker.run(batch_dists_tensor, gallery_target_tensor)

            rank_label_list = batch_rank_label_list[0]
            sort_idx_list = batch_sort_idx_list[0]
            rank_img_name_list = list(np.array(gallery_img_name_list)[sort_idx_list])
            assert len(rank_label_list) == len(rank_img_name_list)

            rank_list = [[rank_img_name, rank_label] for rank_img_name, rank_label in
                         zip(rank_img_name_list[:self.top_k], rank_label_list[:self.top_k])]

            save_path = os.path.join(self.save_dir, f'{query_img_name}.csv')
            np.savetxt(save_path, np.array(rank_list, dtype=object), fmt='%s', delimiter=KEY_SEP)
            content_dict[query_img_name] = query_label

        info_dict = {
            'classes': query_cls_list,
            'content': content_dict,
            'query_dir': self.query_dir,
            'gallery_dir': self.gallery_dir
        }
        info_path = os.path.join(self.save_dir, 'info.pkl')
        logger.info(f'save to {info_path}')
        with open(info_path, 'wb') as f:
            pickle.dump(info_dict, f)
