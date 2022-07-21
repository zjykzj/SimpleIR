# -*- coding: utf-8 -*-

"""
@date: 2022/3/28 下午3:21
@file: utils.py
@author: zj
@description: 
"""

from typing import Dict

import os
import glob
import torch
import pickle

from zcls2.util import logging

logger = logging.get_logger(__name__)


def save_model(model, dst_root='outputs', epoch=0):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    assert os.path.isdir(dst_root), dst_root

    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    model_path = os.path.join(dst_root, f'model_e{epoch}.pt')
    torch.save(state, model_path)


def load_model(model, model_path, device=torch.device('cpu')):
    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError(f'There is no key `state_dict` in {model_path}')

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    logger.info("=> loaded checkpoint '{}'".format(model_path, ))


def load_feats(feat_dir: str, prefix='part_') -> Dict:
    assert os.path.isdir(feat_dir), feat_dir

    gallery_dict = dict()

    file_list = glob.glob(os.path.join(feat_dir, f'{prefix}*.csv'))
    for file_path in file_list:
        with open(file_path, 'rb') as f:
            tmp_feats_list = pickle.load(f)['feats']

            for idx, tmp_feat_dict in enumerate(tmp_feats_list):
                tmp_feat = tmp_feat_dict['feat']
                tmp_label = tmp_feat_dict['label']

                if tmp_label not in gallery_dict.keys():
                    gallery_dict[tmp_label] = list()
                gallery_dict[tmp_label].append(torch.from_numpy(tmp_feat))

    return gallery_dict
