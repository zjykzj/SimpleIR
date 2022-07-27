# -*- coding: utf-8 -*-

"""
@date: 2022/5/11 下午10:40
@file: mobilenet.py
@author: zj
@description: Custom MobileNetV3, derived from torchvision
"""
from typing import Any, List, Optional, Callable, Dict
from functools import partial

from torch import Tensor, nn
from torchvision.models.mobilenetv3 import MobileNetV3 as TMobileNetV3
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, model_urls, \
    InvertedResidualConfig

from zcls2.model.model.mobilenet import load_state_dict_from_url
from zcls2.config.key_word import KEY_OUTPUT
from simpleir.configs.key_words import KEY_FEAT
from simpleir.models.model_base import ModelBase

__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


class MobileNetV3(TMobileNetV3, ModelBase):
    _feat_list = [
        'blocks', 'avgpool', 'linear', 'hardswish', 'classifier'
    ]

    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int,
                 num_classes: int = 1000, block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 feat_type='hardswish', **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer, **kwargs)
        assert feat_type in self._feat_list

        self.feature_modules = {
            "blocks": self.features[16][2],
            "avgpool": self.avgpool,
            "linear": self.classifier[0],
            "hardswish": self.classifier[1],
            "classifier": self.classifier[3],
        }
        self.feature_buffer = dict()
        self.feat_type = feat_type
        self._register_hook()

    def forward(self, x: Tensor) -> Dict:
        x = super().forward(x)
        feat = self.feature_buffer[self.feat_type]

        return {
            KEY_OUTPUT: x,
            KEY_FEAT: feat
        }


def _mobilenet_v3_model(
        arch: str,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        # If the number of model outputs is different from the model setting,
        # the corresponding pretraining model weight will not be loaded
        assert isinstance(model.classifier[3], nn.Linear)
        if model.classifier[3].out_features != 1000:
            state_dict.pop('classifier.3.weight')
            state_dict.pop('classifier.3.bias')

        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {'classifier.3.weight', 'classifier.3.bias'}, \
            f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
