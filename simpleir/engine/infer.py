# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: infer.py
@author: zj
@description: 
"""

from typing import List, Dict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from zcls2.util import logging
from zcls2.util.meter import AverageMeter
from zcls2.util.misc import to_python_float

logger = logging.get_logger(__name__)

from simpleir.configs.key_words import KEY_FEAT
from simpleir.metric.helper import MetricHelper
from simpleir.utils.prefetcher import data_prefetcher


def validate(cfg: CfgNode, val_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> List:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_k = cfg.TRAIN.TOP_K
    top_list = [AverageMeter() for _ in top_k]

    # switch to evaluate mode
    model.eval()
    end = time.time()

    aggregate_type = cfg.METRIC.AGGREGATE_TYPE
    enhance_type = cfg.METRIC.ENHANCE_TYPE
    distance_type = cfg.METRIC.DISTANCE_TYPE
    rank_type = cfg.METRIC.RANK_TYPE
    re_rank_type = cfg.METRIC.RE_RANK_TYPE
    metric = MetricHelper(cfg.METRIC.MAX_CATE_NUM,
                          aggregate_type=aggregate_type,
                          enhance_type=enhance_type,
                          top_k_list=top_k, distance_type=distance_type,
                          rank_type=rank_type, re_rank_type=re_rank_type)

    prefetcher = data_prefetcher(cfg, val_loader)
    input, target, paths = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        # prec_list = metric.run(output[KEY_FEAT].detach().cpu().numpy(), target.detach().cpu().numpy(),
        prec_list = metric.run(output[KEY_FEAT].detach().cpu(), target.detach().cpu())

        reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        for idx, prec in enumerate(prec_list):
            top_list[idx].update(prec, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        # if cfg.RANK_ID == 0 and i % cfg.PRINT_FREQ == 0:
        if cfg.RANK_ID == 0:
            logger_str = 'Test: [{0}/{1}] ' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                         'Speed {2:.3f} ({3:.3f}) ' \
                         'Loss {loss.val:.4f} ({loss.avg:.4f}) '.format(
                i, len(val_loader),
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.val,
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.avg,
                batch_time=batch_time, loss=losses)
            for k, top in zip(top_k, top_list):
                logger_str += f'Prec@{k} {top.val:.3f} ({top.avg:.3f}) '
            logger.info(logger_str)

        input, target, paths = prefetcher.next()

    metric.clear()

    logger_str = ' * '
    for k, top in zip(top_k, top_list):
        logger_str += f'Prec@{k} {top.avg:.3f} '
    logger.info(logger_str)

    return [top.avg for top in top_list]
