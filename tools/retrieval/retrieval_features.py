# -*- coding: utf-8 -*-

"""
@date: 2022/7/16 下午3:44
@file: retrieval_features.py
@author: zj
@description: Retrieval features. You should set --query-dir and --gallery-dir. The results will save in --save-dir
for example,
1. python retrieval_features.py --query-dir data/query_fc --gallery-dir data/gallery_fc --save-dir data/retrieval_fc
2. python retrieval_features.py --query-dir data/query_fc --gallery-dir data/gallery_fc --save-dir data/retrieval_fc --topk 20

You can find info.pkl in save-dir at the end of the program. It's a dict and save like this:
1. 'classes': [cls1, cls2, ...]
2. 'content': {query_name: label, query_name2: label2, ...},
2. 'query_dir': args.query_dir,
3. 'gallery_dir': args.gallery_dir
"""

import os
import argparse
import pickle

from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from typing import List

dis_list = ['euclidean', 'cosine']

retrieval_list = ['sort', 'knn']


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval features")
    parser.add_argument('--query-dir', metavar='QUERY', default=None, type=str,
                        help='Dir for loading query features. Default: None')
    parser.add_argument('--gallery-dir', metavar='GALLERY', default=None, type=str,
                        help='Dir for loading gallery features. Default: None')

    parser.add_argument('--distance', metavar='DISTANCE', default='euclidean', type=str, choices=dis_list,
                        help='The way to compute distance. Default: euclidean')
    parser.add_argument('--retrieval', metavar='RETRIEVAL', default='sort', type=str, choices=retrieval_list,
                        help='The way to retrieval. Default: srot')

    parser.add_argument('--save-dir', metavar='SAVE', default=None, type=str,
                        help='Dir for saving retrieval results. Default: None')
    parser.add_argument('--topk', metavar='TOPK', default=None, type=int,
                        help='Saving topk results. Default: None (Save all)')

    return parser.parse_args()


def load_features(feat_dir):
    assert os.path.isdir(feat_dir), feat_dir

    info_path = os.path.join(feat_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    feat_list = list()
    label_list = list()
    img_name_list = list()
    for img_name, label in tqdm(info_dict['content'].items()):
        feat_path = os.path.join(feat_dir, f'{img_name}.npy')
        feat = np.load(feat_path)

        feat_list.append(feat)
        label_list.append(label)
        img_name_list.append(img_name)

    return feat_list, label_list, info_dict['classes'], img_name_list


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
    refer to [TORCH.CDIST](https://pytorch.org/docs/stable/generated/torch.cdist.html)

    torch.cdist(B×P×M, B×R×M) -> (BxPxR)
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]

    res = torch.cdist(x1, x2, p=2)
    return res


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Calculate the distance between query set features and gallery set features.

    Args:
        x1 (torch.tensor): query set features.
        x2 (torch.tensor): gallery set features.

    Returns:
        dis (torch.tensor): the cosine distance between query set features and gallery set features.
    """
    assert len(x1.shape) == len(x2.shape) == 2 and x1.shape[1] == x2.shape[1]
    similarity_matrix = F.cosine_similarity(x1.unsqueeze(1),
                                            x2.unsqueeze(0), dim=2)

    return 1 - similarity_matrix


def argsort(data: torch.Tensor) -> torch.Tensor:
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    return torch.argsort(data, dim=1)


def normal_rank(batch_sorts: Tensor, gallery_targets: Tensor) -> List[List]:
    rank_list = list()
    for sort_arr in batch_sorts:
        sorted_list = gallery_targets[sort_arr].int().tolist()

        rank_list.append(sorted_list)

    return rank_list


def process(query_feat_tensor: Tensor, gallery_feat_tensor: Tensor, gallery_target_tensor: Tensor,
            distance='euclidean', retrieval='sort'):
    """
    Calculate the similarity between the query feature and the search set feature, sort, and return the sorted label
    """
    if distance == 'euclidean':
        batch_dists = euclidean_distance(query_feat_tensor, gallery_feat_tensor)
    elif distance == 'cosine':
        batch_dists = cosine_distance(query_feat_tensor, gallery_feat_tensor)
    else:
        raise ValueError(f'{distance} does not support.')

    if retrieval == 'sort':
        # The more smaller distance, the more similar object
        batch_sorts = argsort(batch_dists)
        batch_ranks = normal_rank(batch_sorts, gallery_target_tensor)
    else:
        raise ValueError(f'{retrieval} does not support')

    return batch_ranks[0]


def main():
    args = parse_args()
    print('Args:', args)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f'Save query features from {args.query_dir}')
    query_feat_list, query_label_list, query_cls_list, query_name_list = load_features(args.query_dir)
    print(f'Save gallery features from {args.gallery_dir}')
    gallery_feat_list, gallery_label_list, gallery_cls_list, _ = load_features(args.gallery_dir)
    assert query_cls_list == gallery_cls_list

    # Retrieval features
    print('Batch process ...')
    content_dict = OrderedDict()
    topk = args.topk
    assert topk is None or (topk > 0 and topk <= len(query_feat_list))

    gallery_feat_tensor = torch.from_numpy(np.array(gallery_feat_list))
    gallery_target_tensor = torch.from_numpy(np.array(gallery_label_list))
    for query_feat, query_label, query_name in tqdm(zip(query_feat_list, query_label_list, query_name_list)):
        tmp_query_feat_list = [query_feat]
        query_feat_tensor = torch.from_numpy(np.array(tmp_query_feat_list))

        rank_label_list = process(query_feat_tensor, gallery_feat_tensor, gallery_target_tensor,
                                  distance=args.distance, retrieval=args.retrieval)
        # print(rank_label_list)

        save_path = os.path.join(save_dir, f'{query_name}.csv')
        np.savetxt(save_path, np.array(rank_label_list)[:topk], fmt='%d', delimiter=' ')
        content_dict[query_name] = query_label

    info_dict = {
        'classes': query_cls_list,
        'content': content_dict,
        'query_dir': args.query_dir,
        'gallery_dir': args.gallery_dir
    }
    info_path = os.path.join(save_dir, 'info.pkl')
    print(f'save to {info_path}')
    with open(info_path, 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == '__main__':
    main()
