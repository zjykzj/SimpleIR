# -*- coding: utf-8 -*-

"""
@date: 2022/5/23 下午2:43
@file: general_dataset.py
@author: zj
@description: 
"""
from typing import Any, Optional, Callable

import torch
from torchvision.datasets.folder import default_loader
from zcls2.data.dataset import general_dataset

__all__ = ['GeneralDataset', 'General']


class GeneralDataset(general_dataset.GeneralDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, w_path: bool = False) -> None:
        super().__init__(root, transform, target_transform)

        self.w_path = w_path

    def __getitem__(self, index: int) -> Any:
        img_path = self.data_list[index]
        target = self.target_list[index]

        image = default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.w_path:
            return image, torch.tensor(int(target)), img_path
        else:
            return image, torch.tensor(int(target))

    def __len__(self) -> int:
        return super().__len__()


General = GeneralDataset
