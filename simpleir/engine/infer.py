# -*- coding: utf-8 -*-

"""
@date: 2022/4/3 下午1:40
@file: infer.py
@author: zj
@description: 
"""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode
# from zcls2.config.key_word import KEY_OUTPUT
from zcls2.util import logging
# from zcls2.util.distributed import reduce_tensor
from zcls2.util.meter import AverageMeter
from zcls2.util.misc import to_python_float
from zcls2.util.prefetcher import data_prefetcher

logger = logging.get_logger(__name__)

from simpleir.configs.key_words import KEY_FEAT


def similarity(feat, gallery_dict: dict):
    """
    计算欧式距离相似度，返回排序最高的前10个结果
    """
    sim_list = list()
    for idx, (key, values) in enumerate(gallery_dict.items()):
        if len(values) == 0:
            continue
        tmp_array = np.linalg.norm(feat - values, axis=1)
        sim_list.extend([[key, score] for score in tmp_array])
    if len(sim_list) == 0:
        return None
    sim_array = np.array(sim_list)
    if len(sim_array) == 1:
        tmp_top_list = [int(sim_array[0][0])]
    else:
        # sort_array = np.argsort(sim_array[:, 1])[::-1]
        sort_array = np.argsort(sim_array[:, 1])
        tmp_top_list = list(sim_array[:, 0][sort_array].astype(int))

    top_list = [0 for _ in range(10)]
    if len(tmp_top_list) < 10:
        top_list[:len(tmp_top_list)] = tmp_top_list[:]
    else:
        top_list = tmp_top_list[:10]

    return top_list


def gallery_accuracy(feats, targets, gallery_dict, topk=(1, 5)):
    top1 = 0
    top5 = 0
    for feat, target in zip(feats, targets):
        # 将特征向量拉平为一维向量
        feat = feat.reshape(-1)
        truth_key = int(target)
        top_10_list = similarity(feat, gallery_dict)
        if top_10_list is None:
            pass
        else:
            if truth_key == top_10_list[0]:
                top1 += 1
            if truth_key in top_10_list[:5]:
                top5 += 1

        # 每次都将feat加入图集，如果该类别保存已满，那么弹出最开始加入的数据
        if truth_key not in gallery_dict.keys():
            gallery_dict[truth_key] = list()
        if len(gallery_dict[truth_key]) > 20:
            gallery_dict[truth_key].pop(0)
        gallery_dict[truth_key].append(feat)

    total_num = len(feats)
    prec1 = 100.0 * top1 / total_num
    prec5 = 100.0 * top5 / total_num
    return prec1, prec5


def validate(cfg: CfgNode, val_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> Tuple[float, float]:
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # 特征集，每个类别保存20条特征，先进先出
    gallery_dict = dict()

    prefetcher = data_prefetcher(cfg, val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output[KEY_OUTPUT].data, target, topk=(1, 5))
        # 假定使用输出作为特征向量
        prec1, prec5 = gallery_accuracy(output[KEY_FEAT].detach().cpu().numpy(), target.detach().cpu().numpy(),
                                        gallery_dict, topk=(1, 5))

        # if cfg.DISTRIBUTED:
        #     reduced_loss = reduce_tensor(cfg.NUM_GPUS, loss.data)
        #     prec1 = reduce_tensor(cfg.NUM_GPUS, prec1)
        #     prec5 = reduce_tensor(cfg.NUM_GPUS, prec5)
        # else:
        reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        # top1.update(to_python_float(prec1), input.size(0))
        # top5.update(to_python_float(prec5), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if cfg.RANK_ID == 0 and i % cfg.PRINT_FREQ == 0:
            logger.info('Test: [{0}/{1}] '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Speed {2:.3f} ({3:.3f}) '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader),
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.val,
                cfg.NUM_GPUS * cfg.DATALOADER.TRAIN_BATCH_SIZE / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        input, target = prefetcher.next()

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg, top5.avg
