# -*- coding: utf-8 -*-

"""
@date: 2022/7/19 上午10:04
@file: buld.py
@author: zj
@description:
"""

import os

from yacs.config import CfgNode
from argparse import Namespace

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from simpleir.configs import get_cfg_defaults
from simpleir.models import *
from simpleir.data import build_data
from simpleir.utils.util import load_model

from .helper import ExtractHelper

from zcls2.util import logging

logger = logging.get_logger(__name__)

__all__ = ['build_cfg']


def build_args(args: Namespace) -> ExtractHelper:
    cfg = get_cfg_defaults()
    cfg.MODEL.ARCH = args.model_arch
    cfg.RESUME = args.pretrained
    cfg.RETRIEVAL.EXTRACT.FEAT_TYPE = args.layer

    if args.gallery:
        cfg.RETRIEVAL.EXTRACT.GALLERY_DIR = args.save_dir
        cfg.DATASET.GALLERY_DIR = args.image_dir
    else:
        cfg.RETRIEVAL.EXTRACT.QUERY_DIR = args.save_dir
        cfg.DATASET.QUERY_DIR = args.image_dir
    cfg.DATASET.RETRIEVAL_NAME = args.dataset

    cfg.TRANSFORM.TEST_METHODS = ('Resize', 'CenterCrop', 'ToTensor', 'Normalize')
    cfg.TRANSFORM.TEST_RESIZE = (256,)
    cfg.TRANSFORM.TEST_CROP = (224, 224)
    cfg.TRANSFORM.NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), False, 1.0)

    cfg.RETRIEVAL.EXTRACT.AGGREGATE_TYPE = args.aggregate
    cfg.RETRIEVAL.EXTRACT.ENHANCE_TYPE = args.enhance
    cfg.RETRIEVAL.EXTRACT.LEARN_PCA = args.learn
    cfg.RETRIEVAL.EXTRACT.PCA_PATH = args.pca
    cfg.RETRIEVAL.EXTRACT.REDUCE_DIMENSION = args.rd
    cfg.freeze()

    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(cfg, device)

    # Optionally resume from a checkpoint
    if cfg.RESUME:
        logger.info("=> Resume now")
        load_model(model, cfg.RESUME, device=device)
    model.eval()

    # Data loading
    _, data_loader = build_data(cfg, is_train=False, is_gallery=args.gallery, w_path=True)

    return build_cfg(cfg, model, data_loader, is_gallery=args.gallery, device=device)


def build_cfg(cfg: CfgNode, model: Module, data_loader: DataLoader, is_gallery=True, device=torch.device('cpu')):
    model_arch = cfg.MODEL.ARCH
    pretrained = cfg.RESUME
    layer = cfg.RETRIEVAL.EXTRACT.FEAT_TYPE

    save_dir = cfg.RETRIEVAL.EXTRACT.GALLERY_DIR if is_gallery else cfg.RETRIEVAL.EXTRACT.QUERY_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    aggregate = cfg.RETRIEVAL.EXTRACT.AGGREGATE_TYPE
    enhance = cfg.RETRIEVAL.EXTRACT.ENHANCE_TYPE
    learn_pca = cfg.RETRIEVAL.EXTRACT.LEARN_PCA
    pca_path = cfg.RETRIEVAL.EXTRACT.PCA_PATH
    rd = cfg.RETRIEVAL.EXTRACT.REDUCE_DIMENSION

    extract_helper = ExtractHelper(model=model, device=device,
                                   model_arch=model_arch, pretrained=pretrained, layer=layer,
                                   data_loader=data_loader, save_dir=save_dir,
                                   aggregate_type=aggregate, enhance_type=enhance,
                                   is_gallery=is_gallery,
                                   learn_pca=learn_pca, pca_path=pca_path, reduce_dimension=rd)
    return extract_helper
