# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午10:55
@file: extract_features.py
@author: zj
@description:
1. 加载数据
2. 加载图像处理器
3. 加载模型
4. 提取特征
"""

import os
import torch
import argparse

from simpleir.configs import get_cfg_defaults
from simpleir.data.build import build_data
from simpleir.models.build import build_model
from simpleir.metric.feature.helper import FeatureHelper
from simpleir.utils.extract.helper import ExtractHelper


def parse_args():
    parser = argparse.ArgumentParser(description="Make Query and Gallery Set")
    parser.add_argument('cfg',
                        type=str,
                        default="",
                        metavar="CONFIG",
                        help="path to config file")
    parser.add_argument('-dst', metavar='DST', default='', type=str, help='Path to the save feature.')
    parser.add_argument('-s', '--save-interval', metavar='INTERVAL',
                        default=5000, type=int, help='Save interval. Default: 5000')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('args:', args)
    save_interval = args.save_interval

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    dst_root = os.path.join(cfg.OUTPUT_DIR, 'extract')
    if args.dst != '':
        dst_root = args.dst
    print(f'extract feats to {dst_root}')

    train_sampler, train_loader, val_loader = build_data(cfg, w_path=True)
    device = torch.device(f'cuda:{cfg.RANK_ID}') if cfg.DISTRIBUTED else torch.device('cpu')
    model = build_model(cfg, device)
    model.eval()

    aggregate_type = cfg.METRIC.AGGREGATE_TYPE
    distance_type = cfg.METRIC.ENHANCE_TYPE
    feature_helper = FeatureHelper(aggregate_type=aggregate_type, enhance_type=distance_type)

    print('train ...')
    extractor = ExtractHelper(train_loader, model, feature_helper)
    extractor.run(dst_root, save_prefix='train', save_interval=save_interval)

    print('test ...')
    extractor = ExtractHelper(val_loader, model, feature_helper)
    extractor.run(dst_root, save_prefix='val', save_interval=save_interval)
