# -*- coding: utf-8 -*-

"""
@date: 2022/7/27 下午2:17
@file: metric_base.py
@author: zj
@description: 
"""

import os


def load_retrieval(retrieval_dir: str) -> Tuple[List[List[str]], List[List[int]], List[str], List[int]]:
    assert os.path.isdir(retrieval_dir), retrieval_dir

    info_path = os.path.join(retrieval_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        info_dict = pickle.load(f)

    batch_rank_label_list = list()
    batch_rank_name_list = list()
    query_name_list = list()
    query_label_list = list()
    for idx, (img_name, label) in enumerate(tqdm(info_dict['content'].items())):
        rank_path = os.path.join(retrieval_dir, f'{img_name}.csv')
        rank_array = np.loadtxt(rank_path, dtype=np.str, delimiter=KEY_SEP)

        batch_rank_name_list.append(list(rank_array[:, 0].astype(str)))
        batch_rank_label_list.append(list(rank_array[:, 1].astype(int)))
        query_name_list.append(img_name)
        query_label_list.append(label)

    return batch_rank_name_list, batch_rank_label_list, query_name_list, query_label_list


class MetricBase(object):

    def __init__(self, retrieval_dir, top_k_list=(1, 3, 5, 10)):
        self.retrieval_dir = retrieval_dir
        assert os.path.isdir(self.retrieval_dir), self.retrieval_dir

        self.batch_rank_name_list, self.batch_rank_label_list, self.query_name_list, self.query_label_list = \
            load_retrieval(self.retrieval_dir)
        self.top_k_list = top_k_list

    def run(self):
        pass
