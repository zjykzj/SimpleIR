# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午11:00
@file: helper.py
@author: zj
@description: 图像特征提取辅助类
1. 批量计算图像特征
2. 批量保存图像特征

每条图像路径对应图像特征
"""

import os
import time
import torch
import pickle

from tqdm import tqdm
from yacs.config import CfgNode

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from zcls2.data.dataloader.collate import fast_collate
from zcls2.util import logging

logger = logging.get_logger(__name__)

from simpleir.configs.key_words import KEY_FEAT
from simpleir.data.build import build_transform
from simpleir.data.build import build_dataset
from simpleir.models.build import build_model
from simpleir.eval.feature.helper import FeatureHelper
from simpleir.utils.prefetcher import data_prefetcher


def save_part_feat(feat_dict, part_file_path) -> None:
    assert not os.path.isfile(part_file_path), part_file_path

    file_dir = os.path.split(part_file_path)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(part_file_path, 'wb') as f:
        pickle.dump(feat_dict, f)


def create_loader(cfg: CfgNode, is_gallery: bool = False) -> DataLoader:
    val_transform, val_target_transform = build_transform(cfg, is_train=False)
    data_set = build_dataset(cfg, val_transform, val_target_transform, is_train=is_gallery, w_path=True)

    test_batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

    if cfg.CHANNELS_LAST:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    if cfg.DATALOADER.COLLATE_FN == 'fast':
        collate_fn = lambda b: fast_collate(b, memory_format)
    else:
        collate_fn = default_collate

    # Ensure the consistency of output sequence. Set shuffle=False and num_workers=0
    val_loader = DataLoader(
        data_set,
        batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        sampler=None,
        collate_fn=collate_fn)

    return val_loader


def load_model(model, model_path, device=torch.device('cpu')) -> None:
    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError(f'There is no key `state_dict` in {model_path}')

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logger.info("=> loaded checkpoint '{}'".format(model_path, ))


class Extractor:
    """
    A helper class to extract feature maps from model, and then aggregate them.
    """

    def __init__(self, cfg: CfgNode, is_gallery: bool = False) -> None:
        # Load data / model / feature_helper
        val_loader = create_loader(cfg, is_gallery=is_gallery)

        device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = build_model(cfg, device)

        # Optionally resume from a checkpoint
        if os.path.isfile(cfg.RESUME):
            logger.info(f"=> Resume now")
            load_model(model, cfg.RESUME, device)
        model.eval()

        aggregate_type = cfg.EVAL.FEATURE.AGGREGATE_TYPE
        distance_type = cfg.EVAL.FEATURE.ENHANCE_TYPE
        feature_helper = FeatureHelper(aggregate_type=aggregate_type, enhance_type=distance_type)

        self.cfg = cfg
        self.data_loader = val_loader
        self.model = model
        self.feature = feature_helper
        self.device = device

        self.classes = self.data_loader.dataset.classes

    def run(self, dst_root, save_prefix: str = 'part_', save_interval: int = 5000) -> None:
        feat_dict = dict()
        feat_dict['classes'] = self.classes
        feat_dict['feats'] = list()
        feat_num = 0

        part_count = 0
        start = time.time()
        prefetcher = data_prefetcher(self.cfg, self.data_loader)

        for images, targets, paths in tqdm(prefetcher):
            if images is None:
                break

            # 提取特征
            feats = self.model(images.to(self.device))[KEY_FEAT].detach().cpu()
            new_feats = self.feature.run(feats)

            for path, target, feat in zip(paths, targets.detach().cpu().numpy(), new_feats.numpy()):
                feat_dict['feats'].append({
                    'path': path,
                    'label': target,
                    'feat': feat
                })
                feat_num += 1
            if feat_num > save_interval:
                save_part_feat(feat_dict, os.path.join(dst_root, f'{save_prefix}{part_count}.pkl'))
                part_count += 1

                del feat_dict
                feat_dict = dict()
                feat_dict['classes'] = self.classes
                feat_dict['feats'] = list()
                feat_num = 0
        if feat_num > 1:
            save_part_feat(feat_dict, os.path.join(dst_root, f'{save_prefix}{part_count}.pkl'))
        end = time.time()
        print('time: ', end - start)
