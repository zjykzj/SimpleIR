# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: infer.py
@author: zj
@description: 
"""

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from zcls2.util import logging

logger = logging.get_logger(__name__)

import simpleir.utils.extract as extract
import simpleir.utils.retrieval as retrieval
import simpleir.utils.metric as metric
from simpleir.utils.metric import EvaluateType


def validate(cfg: CfgNode, model: nn.Module, query_loader: DataLoader, gallery_loader: DataLoader,
             device=torch.device('cpu')) -> List[float]:
    # switch to evaluate mode
    model.eval()

    logger.info("=> Extract gallery images")
    gallery_extract_helper = extract.build_cfg(cfg, model, gallery_loader, is_gallery=True, device=device)
    gallery_extract_helper.run()

    logger.info("=> Extract query images")
    query_extract_helper = extract.build_cfg(cfg, model, query_loader, is_gallery=False, device=device)
    query_extract_helper.run()

    logger.info("=> Retrieval features")
    retrieval_helper = retrieval.build_cfg(cfg)
    retrieval_helper.run()

    logger.info("=> Evaluate features")
    metric_helper = metric.build_cfg(cfg)
    top_list = metric_helper.run()

    retrieval_type = cfg.RETRIEVAL.METRIC.EVAL_TYPE
    top_k_list = cfg.RETRIEVAL.METRIC.TOP_K
    logger_str = ' * '
    if retrieval_type == EvaluateType.MAP_OXFORD.value:
        logger_str += f'MAP {top_list[0]:.3f}% '
    else:
        for k, top in zip(top_k_list, top_list):
            if retrieval_type == EvaluateType.ACCURACY.value:
                logger_str += f'Acc@{k} {top:.3f}% '
            elif retrieval_type == EvaluateType.PRECISION.value:
                logger_str += f'Prec@{k} {top:.3f}% '
            else:
                logger_str += f'MAP@{k} {top:.3f}% '
    logger.info(logger_str)

    return top_list
