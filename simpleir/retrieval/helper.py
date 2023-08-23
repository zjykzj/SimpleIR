# -*- coding: utf-8 -*-

"""
@date: 2022/5/16 下午2:52
@file: Retriever.py
@author: zj
@description: 
"""
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import torch

from .impl.distancer import DistanceType, do_distance
from .impl.ranker import RankType, do_rank
from .impl.reranker import ReRankType
from ..utils.logger import LOGGER
from ..utils.misc import colorstr

__all__ = ['RetrievalHelper']


class RetrievalHelper:
    """
    图像检索需要什么？

    1. 图像特征
    2. 图像标签
    3. 更好的展示目的，可以有图像类别

    需要重新读取编译吗？在特征提取阶段，这些数据都应该已经存在了，剩下的就是把数据传递进来

    保存每次检索得到的检索结果，包括检索图像名、检索排序后的前top_k个结果

    反馈图像名和对应的标签？？？都可以
    """

    def __init__(self,
                 distance_type: str = 'EUCLIDEAN',
                 rank_type: str = 'NORMAL',
                 knn_top_k: int = 5,
                 rerank_type: str = 'IDENTITY',
                 ):
        self.distance_type = DistanceType[distance_type]
        self.rank_type = RankType[rank_type]
        self.knn_top_k = knn_top_k
        self.rerank_type = ReRankType[rerank_type]

    def run(self, gallery_img_name_list, gallery_feat_list, gallery_label_list,
            query_img_name_list, query_feat_list, query_label_list):
        gallery_feat_tensor = torch.from_numpy(np.array(gallery_feat_list, dtype=np.float32))
        gallery_label_tensor = torch.from_numpy(np.array(gallery_label_list, dtype=int))

        content_dict = OrderedDict()
        for query_img_name, query_feat, query_label in \
                tqdm(zip(query_img_name_list, query_feat_list, query_label_list), total=len(query_feat_list)):
            query_feat_tensor = torch.from_numpy(np.array(query_feat)).unsqueeze(0)

            batch_dists_tensor = do_distance(query_feat_tensor, gallery_feat_tensor, self.distance_type)
            batch_sort_idx_list, batch_rank_label_list = \
                do_rank(batch_dists_tensor, gallery_label_tensor, self.rank_type, self.knn_top_k)

            rank_label_list = batch_rank_label_list[0]
            sort_idx_list = batch_sort_idx_list[0]
            rank_img_name_list = list(np.array(gallery_img_name_list)[sort_idx_list])
            assert len(rank_label_list) == len(rank_img_name_list)

            content_dict[query_img_name] = [query_label, rank_img_name_list, rank_label_list]

        return content_dict
