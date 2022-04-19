# -*- coding: utf-8 -*-

"""
@date: 2022/4/19 上午11:00
@file: build.py
@author: zj
@description: 构建提取器
1. 输入数据集类型和数据集根路径
2. 创建数据集类、图像转换器、数据集加载器
3. 创建模型
4. 依次遍历图像，保存图像提取的特征
"""

from typing import Tuple, Any

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models

from .helper import ExtractHelper


class CustomImageFolder(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def create_transform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return transform


def build_model():
    return models.resnet50(pretrained=True)


def build_extractor(data_root, dataset_type):
    transform = create_transform()
    dataset = CustomImageFolder(data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, sampler=None, num_workers=4, pin_memory=True)

    model = build_model()

    helper = ExtractHelper(dataloader, model)
    return helper
