# -*- coding: utf-8 -*-

"""
@date: 2022/4/25 下午4:08
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from yacs.config import CfgNode

from zcls2.util import logging

logger = logging.get_logger(__name__)

from . import tiny_autocoder, resnet, ghostnet, mobilenet

__all__ = ["build_model"]


def build_model(cfg: CfgNode, device: torch.device = torch.device('cpu')) -> nn.Module:
    model_arch = cfg.MODEL.ARCH
    is_pretrained = cfg.MODEL.PRETRAINED
    num_classes = cfg.MODEL.NUM_CLASSES
    sync_bn = cfg.MODEL.SYNC_BN

    feat_type = cfg.EVAL.FEATURE.FEAT_TYPE

    # create model
    if is_pretrained:
        logger.info("=> using pre-trained model '{}'".format(model_arch))
    else:
        logger.info("=> creating model '{}'".format(model_arch))

    if model_arch in tiny_autocoder.__all__:
        model = tiny_autocoder.__dict__[model_arch]()
    elif model_arch in resnet.__all__:
        model = resnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes, feat_type=feat_type)
    elif model_arch in ghostnet.__all__:
        model = ghostnet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes, feat_type=feat_type)
    elif model_arch in mobilenet.__all__:
        model = mobilenet.__dict__[model_arch](pretrained=is_pretrained, num_classes=num_classes, feat_type=feat_type)
    else:
        raise ValueError(f"{model_arch} does not support")

    if sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if cfg.CHANNELS_LAST:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    # Same as Apex setting
    model = model.to(device, memory_format=memory_format)

    return model
