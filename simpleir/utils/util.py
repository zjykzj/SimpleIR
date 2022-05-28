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


def load_model(model, model_path):
    assert isinstance(model, torch.nn.Module)
    assert os.path.isfile(model_path), model_path

    state = torch.load(model_path)
    model.load_state_dict(state['net'])

    return model


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
