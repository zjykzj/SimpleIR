# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:33
@file: map_for_oxford.py
@author: zj
@description: 
"""
from typing import List, Set, Dict

import os
import glob

import numpy as np

from ..metric_base import MetricBase

__all__ = ["MapForOxford"]


class MapForOxford(MetricBase):
    """
    See https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
    """

    def __init__(self, data_root: str, retrieval_dir: str):
        super().__init__(retrieval_dir, None)

        self.data_root = data_root

    def get_query_prefix(self, gt_dir: str) -> Dict:
        assert os.path.isdir(gt_dir), gt_dir

        query_file_path_list = glob.glob(os.path.join(gt_dir, '*_query.txt'))

        query_prefix_dict = dict()
        for query_file_path in query_file_path_list:
            with open(query_file_path, 'r') as f:
                img_name = f.readline().strip().split(' ')[0]
                assert img_name.startswith('oxc1_')

                img_name = img_name.replace('oxc1_', '')
                query_prefix_dict[img_name] = query_file_path.replace('_query.txt', '')

        return query_prefix_dict

    def load_list(self, file_path: str) -> List:
        assert os.path.isfile(file_path), file_path

        name_list = list()
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() != '':
                    name_list.append(line.strip())

        return name_list

    def compute_ap(self, pos_set: Set[str], junk_set: Set[str], rank_name_list: List[str]) -> float:
        old_recall = 0.
        old_precision = 0.
        ap = 0.

        intersect_size = 0
        j = 0
        for idx, rank_name in enumerate(rank_name_list):
            if rank_name in junk_set:
                continue
            if rank_name in pos_set:
                intersect_size += 1

            recall = intersect_size * 1.0 / len(pos_set)
            precision = intersect_size / (j + 1)

            ap += (recall - old_recall) * ((old_precision + precision) / 2.0)

            old_recall = recall
            old_precision = precision

            j += 1

        return ap

    def compute_map(self, query_prefix_list: List[str], batch_rank_name_list: List[List[str]]) -> List[float]:
        map = 0.
        for query_prefix, rank_name_list in zip(query_prefix_list, batch_rank_name_list):
            good_set = set(self.load_list(f"{query_prefix}_good.txt"))
            ok_set = set(self.load_list(f"{query_prefix}_ok.txt"))
            junk_set = set(self.load_list(f"{query_prefix}_junk.txt"))

            pos_set = set(list(good_set) + list(ok_set))
            ap = self.compute_ap(pos_set, junk_set, rank_name_list)

            map += ap

        return [map * 100.0 / len(query_prefix_list)]

    def run(self):
        query_prefix_dict = self.get_query_prefix(os.path.join(self.data_root, 'groundtruth'))

        query_prefix_list = list()
        for query_name in self.query_name_list:
            query_prefix_list.append(query_prefix_dict[query_name])

        return self.compute_map(query_prefix_list, self.batch_rank_name_list)
