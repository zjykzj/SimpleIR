# -*- coding: utf-8 -*-

"""
@date: 2022/5/9 下午3:31
@file: r_mac.py
@author: zj
@description: 
"""
from typing import List

cached_regions = dict()


def get_regions(h: int, w: int, level_n: int = 3) -> List:
    """
    Divide the image into several regions.

    Args:
        h (int): height for dividing regions.
        w (int): width for dividing regions.

    Returns:
        regions (List): a list of region positions.
    """
    if (h, w) in cached_regions:
        return cached_regions[(h, w)]

    m = 1
    n_h, n_w = 1, 1
    regions = list()
    if h != w:
        min_edge = min(h, w)
        left_space = max(h, w) - min(h, w)
        iou_target = 0.4
        iou_best = 1.0
        while True:
            iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

            # small m maybe result in non-overlap
            if iou_tmp <= 0:
                m += 1
                continue

            if abs(iou_tmp - iou_target) <= iou_best:
                iou_best = abs(iou_tmp - iou_target)
                m += 1
            else:
                break
        if h < w:
            n_w = m
        else:
            n_h = m

    for i in range(level_n):
        region_width = int(2 * 1.0 / (i + 2) * min(h, w))
        step_size_h = (h - region_width) // n_h
        step_size_w = (w - region_width) // n_w

        for x in range(n_h):
            for y in range(n_w):
                st_x = step_size_h * x
                ed_x = st_x + region_width - 1
                assert ed_x < h
                st_y = step_size_w * y
                ed_y = st_y + region_width - 1
                assert ed_y < w
                regions.append((st_x, st_y, ed_x, ed_y))

        n_h += 1
        n_w += 1

    cached_regions[(h, w)] = regions
    return regions
