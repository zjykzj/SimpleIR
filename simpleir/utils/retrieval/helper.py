# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: Retriever.py
@author: zj
@description: 
"""

import os
import pickle

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch

from simpleir.utils.retrieval.impl.distancer import Distancer
from simpleir.utils.retrieval.impl.ranker import Ranker
from simpleir.utils.retrieval.impl.reranker import ReRanker

from zcls2.config.key_word import KEY_SEP
from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['RetrievalHelper']


def load_features(feat_dir: str):
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

        feat_list.append(feat)
        label_list.append(label)
        img_name_list.append(img_name)

    return feat_list, label_list, info_dict['classes'], img_name_list


class RetrievalHelper:

    def __init__(self, query_dir: str, gallery_dir: str, save_dir: str, topk=None,
                 distance_type: str = 'EUCLIDEAN', rank_type: str = 'NORMAL', re_rank_type='IDENTITY',
                 ):
        self.query_dir = query_dir
        assert os.path.isdir(self.query_dir), self.query_dir
        self.gallery_dir = gallery_dir
        assert os.path.isdir(self.gallery_dir), self.gallery_dir

        self.save_dir = save_dir
        assert os.path.isdir(self.save_dir), self.save_dir
        self.topk = topk

        self.distancer = Distancer(distance_type)
        self.ranker = Ranker(rank_type)
        self.reranker = ReRanker(re_rank_type)

    def run(self):
        logger.info(f"Loading query features from {self.query_dir}")
        query_feat_list, query_label_list, query_cls_list, query_name_list = load_features(self.query_dir)
        logger.info(f"Loading query features from {self.gallery_dir}")
        gallery_feat_list, gallery_label_list, gallery_cls_list, gallery_name_list = load_features(self.gallery_dir)
        assert query_cls_list == gallery_cls_list

        gallery_feat_tensor = torch.from_numpy(np.array(gallery_feat_list))
        gallery_target_tensor = torch.from_numpy(np.array(gallery_label_list))

        logger.info('Retrieval ...')
        content_dict = OrderedDict()
        assert self.topk is None or (0 < self.topk <= len(query_feat_list))

        for query_feat, query_label, query_name in tqdm(zip(query_feat_list, query_label_list, query_name_list)):
            tmp_query_feat_list = [query_feat]
            query_feat_tensor = torch.from_numpy(np.array(tmp_query_feat_list))

            batch_dists_tensor = self.distancer.run(query_feat_tensor, gallery_feat_tensor)

            batch_sorts, rank_label_list = self.ranker.run(batch_dists_tensor, gallery_target_tensor)
            rank_name_list = list(np.array(gallery_name_list)[tuple(rank_label_list)])

            rank_list = [[name, label] for name, label in zip(rank_name_list[:self.topk], rank_label_list[:self.topk])]

            save_path = os.path.join(self.save_dir, f'{query_name}.csv')
            np.savetxt(save_path, np.array(rank_list, dtype=object), fmt='%s', delimiter=KEY_SEP)
            content_dict[query_name] = query_label

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
