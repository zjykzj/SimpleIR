# -*- coding: utf-8 -*-

"""
@date: 2023/8/21 上午11:17
@file: retrieval.py
@author: zj
@description:

Usage - Retrieval Features:
    $ python retrieval.py /path/to/gallery.pkl /path/to/query.pkl

Usage - Use Cosine distance:
    $ python retrieval.py --distance COSINE /path/to/gallery.pkl /path/to/query.pkl

Usage - Use KNN Rank:
    $ python retrieval.py --rank KNN /path/to/gallery.pkl /path/to/query.pkl

"""

import os
import sys

import argparse
from argparse import Namespace

from pathlib import Path

from simpleir.retrieval.helper import RetrievalHelper, DistanceType, RankType, ReRankType
from simpleir.utils.logger import LOGGER
from simpleir.utils.misc import print_args, colorstr
from simpleir.utils.fileutil import increment_path
from simpleir.utils.general import load_features, save_retrieval

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    distance_types = [e.value for e in DistanceType]
    rank_types = [e.value for e in RankType]
    rerank_types = [e.value for e in ReRankType]
    # print(distance_types)
    # print(rank_types)
    # print(rerank_types)

    parser = argparse.ArgumentParser()
    parser.add_argument('gallery', type=str, help='gallery info path')
    parser.add_argument('query', type=str, help='query info path')

    parser.add_argument('--distance', type=str, default='EUCLIDEAN', choices=distance_types,
                        help='distance type: ' +
                             ' | '.join(distance_types) +
                             ' (default: EUCLIDEAN)')
    parser.add_argument('--rank', type=str, default='NORMAL', choices=rank_types,
                        help='rank type: ' +
                             ' | '.join(rank_types) +
                             ' (default: NORMAL)')
    parser.add_argument('--knn-topk', type=int, default=5,
                        help='select the top-k highest similarity lists for knn sorting')
    parser.add_argument('--rerank', type=str, default='IDENTITY', choices=rerank_types,
                        help='rerank type: ' +
                             ' | '.join(rerank_types) +
                             ' (default: IDENTITY)')

    parser.add_argument('--project', default=ROOT / 'runs/retrieval', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()
    return opt


def main(opt: Namespace):
    # Config
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print_args(vars(opt))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Data
    gallery_feat_list, gallery_label_list, gallery_img_name_list, gallery_classes = load_features(opt.gallery)
    query_feat_list, query_label_list, query_img_name_list, query_classes = load_features(opt.query)

    # Retrieval
    retrieval_helper = RetrievalHelper(distance_type=opt.distance,
                                       rank_type=opt.rank,
                                       knn_top_k=opt.knn_topk,
                                       rerank_type=opt.rerank)

    LOGGER.info("Retrieval")
    content_dict = retrieval_helper.run(gallery_img_name_list, gallery_feat_list, gallery_label_list,
                                        query_img_name_list, query_feat_list, query_label_list)

    # Save
    save_retrieval(opt.gallery, opt.query, content_dict, query_classes, opt.save_dir)
    LOGGER.info(f"Save to {colorstr(opt.save_dir)}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
