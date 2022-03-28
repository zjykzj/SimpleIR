# -*- coding: utf-8 -*-

"""
@date: 2022/3/28 下午3:21
@file: utils.py
@author: zj
@description: 
"""

import os
import torch


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