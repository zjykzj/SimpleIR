# -*- coding: utf-8 -*-

"""
@date: 2022/5/11 下午3:32
@file: count.py
@author: zj
@description: Refer to
1. [python-统计元素个数](https://blog.csdn.net/u012991043/article/details/81020086#:~:text=python%E7%BB%9F%E8%AE%A1%E5%88%97%E8%A1%A8%E4%B8%AD%E5%85%83%E7%B4%A0,.count(i)%E3%80%91%E3%80%82)
2. [KNN算法-2-使用Python实现KNN算法](https://www.jianshu.com/p/ce9813d6bff0)
"""

from typing import List

import numpy as np
from collections import Counter


def count_frequency(data_list: List[int]) -> List[int]:
    unique_list = np.unique(data_list)

    res_list = list()
    for u in unique_list:
        res_list.append([u, data_list.count(u)])

    sorted_indices = list(np.argsort(-1 * np.array(res_list)[:, 1]))

    return list(np.array(res_list)[:, 0][sorted_indices])


def count_frequency_v2(data_list: List[int]) -> List[int]:
    res_list = list()
    for u in set(data_list):
        res_list.append([u, data_list.count(u)])

    sorted_indices = list(np.argsort(-1 * np.array(res_list)[:, 1]))

    return list(np.array(res_list)[:, 0][sorted_indices])


def count_frequency_v3(data_list: List[int]) -> List[int]:
    votes = Counter(data_list)

    return list(np.array(votes.most_common())[:, 0])


if __name__ == '__main__':
    a = [1, 3, 1, 6, 1, 3, 1, 6, 13, 3, 5, 61, 1]
    res_list = count_frequency(a)
    print(res_list)

    res_list = count_frequency_v2(a)
    print(res_list)

    res_list = count_frequency_v3(a)
    print(res_list)
