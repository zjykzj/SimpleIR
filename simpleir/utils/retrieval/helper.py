# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 下午8:18
@file: helper.py
@author: zj
@description: 
"""

import os
import glob
import time
import pickle

import torch
from yacs.config import CfgNode

import numpy as np
from zcls2.util.meter import AverageMeter
from zcls2.util import logging

logger = logging.get_logger(__name__)

from simpleir.eval.index.helper import IndexHelper
from simpleir.eval.metric.helper import MetricHelper


class RetrievalHelper:

    def __init__(self, cfg: CfgNode) -> None:
        self.top_k_list = cfg.TRAIN.TOP_K

        distance_type = cfg.EVAL.INDEX.DISTANCE_TYPE
        rank_type = cfg.EVAL.INDEX.RANK_TYPE
        re_rank_type = cfg.EVAL.INDEX.RE_RANK_TYPE
        gallery_dir = cfg.EVAL.INDEX.GALLERY_DIR
        max_num = cfg.EVAL.INDEX.MAX_CATE_NUM
        index_mode = cfg.EVAL.INDEX.MODE
        self.index = IndexHelper(top_k=self.top_k_list[-1],
                                 distance_type=distance_type,
                                 rank_type=rank_type,
                                 re_rank_type=re_rank_type,
                                 gallery_dir=gallery_dir,
                                 max_num=max_num,
                                 index_mode=index_mode)
        self.index.init()

        eval_type = cfg.EVAL.METRIC.EVAL_TYPE
        self.metric = MetricHelper(top_k_list=self.top_k_list, eval_type=eval_type)

        self.query_dir = cfg.EVAL.FEATURE.QUERY_DIR

    def run(self, prefix: str = 'part_') -> None:
        batch_time = AverageMeter()
        top_list = [AverageMeter() for _ in self.top_k_list]

        logger.info(f"Loaded feats from {self.query_dir}")
        file_list = glob.glob(os.path.join(self.query_dir, f'{prefix}*.csv'))
        file_len = len(file_list)

        end = time.time()
        for idx, file_path in enumerate(file_list):
            with open(file_path, 'rb') as f:
                tmp_feats_list = pickle.load(f)['feats']

                query_feats = list()
                query_targets = list()
                for i, tmp_feat_dict in enumerate(tmp_feats_list):
                    tmp_feat = tmp_feat_dict['feat']
                    tmp_label = tmp_feat_dict['label']

                    query_targets.append(tmp_label)
                    query_feats.append(tmp_feat)

            query_feats = torch.from_numpy(np.array(query_feats))
            query_targets = torch.from_numpy(np.array(query_targets, dtype=int))

            pred_top_k_list = self.index.run(query_feats, query_targets)
            prec_list = self.metric.run(pred_top_k_list, query_targets.numpy())

            one_query_len = len(query_feats)
            for i, prec in enumerate(prec_list):
                top_list[i].update(prec, one_query_len)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger_str = 'Retrieval: [{0}/{1}] ' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                         'Speed {2:.3f} ({3:.3f}) '.format(
                idx, file_len,
                one_query_len / batch_time.val, one_query_len / batch_time.avg,
                batch_time=batch_time)
            for k, top in zip(self.top_k_list, top_list):
                logger_str += f'Prec@{k} {top.val:.3f} ({top.avg:.3f}) '
            logger.info(logger_str)

        logger_str = ' * '
        for k, top in zip(self.top_k_list, top_list):
            logger_str += f'Prec@{k} {top.avg:.3f} '
        logger.info(logger_str)

        self.index.clear()
