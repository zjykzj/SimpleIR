# -*- coding: utf-8 -*-

"""
@date: 2022/4/27 下午5:10
@file: rank.py
@author: zj
@description: 
"""
from typing import List

import numpy as np


def normal_rank(similarity_list: List) -> List:
    sim_array = np.array(similarity_list)
    if len(sim_array) == 1:
        top_list = [int(sim_array[0][0])]
    else:
        # The smaller the distance, the more similar
        sort_array = np.argsort(sim_array[:, 1])
        top_list = list(sim_array[:, 0][sort_array].astype(int))

    return top_list


def rank(similarity_list: List, top_k: int = 10, rank_type: str = 'normal') -> List:
    if rank_type == 'normal':
        tmp_top_list = normal_rank(similarity_list)
    else:
        raise ValueError(f'{rank_type} does not support')

    # Returns only the sort results needed to calculate the hit rate
    top_list = [0 for _ in range(top_k)]
    if len(tmp_top_list) < top_k:
        top_list[:len(tmp_top_list)] = tmp_top_list[:]
    else:
        top_list = tmp_top_list[:top_k]

    return top_list
