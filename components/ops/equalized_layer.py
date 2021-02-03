# Code for adding "Equalize Learning Rate" Trick.
# Reference and adapted from:
#  https://github.com/NVlabs/stylegan2
#  https://github.com/rosinality/stylegan2-pytorch
# The hook-based implementation style is adapted from:
#  https://pytorch.org/docs/master/_modules/torch/nn/utils/weight_norm.html#weight_norm
# Author: Xiao Li

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init

import math


class EqualizeLearningRateClass(object):
    r"""
    A class that wraps the layer behavior to:
    - initialize one layer's weight to N(0, 1) and bias to certain values
    - apply He's initialization at runtime (a.k.a "Equalized Learning Rate" trick).
    """
    def __init__(self, weight_name: str, bias_name: Optional[str] = None,
                 coeff_w: float = 1.0, coeff_b: float = 1.0):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.coeff_w = coeff_w
        self.coeff_b = coeff_b

    def scale_weight(self, module):
        real_weight = getattr(module, self.weight_name + '_origin')
        return real_weight * self.coeff_w

    def scale_bias(self, module):
        real_bias = getattr(module, self.bias_name + '_origin')
        return real_bias * self.coeff_b

    @staticmethod
    def apply(module, weight_name: str, bias_name: Optional[str] = None,
              lr_mul: float = 1.0, bias_init: float = 0.0):

        # Setup real weight and its scaled version
        weight = getattr(module, weight_name)
        del module._parameters[weight_name]
        module.register_parameter(weight_name + '_origin', nn.Parameter(torch.zeros_like(weight)))
        init.normal_(getattr(module, weight_name + '_origin'), std=1.0 / lr_mul)

        # Setup real bias and its scaled version
        if(bias_name is not None):
            bias = getattr(module, bias_name)
            del module._parameters[bias_name]
            module.register_parameter(bias_name + '_origin', nn.Parameter(torch.zeros_like(bias).fill_(bias_init)))

        if(hasattr(module, 'transposed') and module.transposed is True):
            fan = init._calculate_correct_fan(weight, 'fan_out')
        else:
            fan = init._calculate_correct_fan(weight, 'fan_in')

        # Compute scale factor
        coeff_w = 1.0 / math.sqrt(fan) * lr_mul
        coeff_b = lr_mul

        fn = EqualizeLearningRateClass(weight_name, bias_name, coeff_w, coeff_b)

        setattr(module, weight_name, fn.scale_weight(module))
        if(bias_name is not None):
            setattr(module, bias_name, fn.scale_bias(module))

        # Recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs):
        setattr(module, self.weight_name, self.scale_weight(module))
        if(self.bias_name is not None):
            setattr(module, self.bias_name, self.scale_bias(module))


def EqualizeLearningRate(module, weight_name: str = 'weight', bias_name: Optional[str] = None,
                         lr_mul: float = 1.0, bias_init: float = 0.0):
    EqualizeLearningRateClass.apply(module, weight_name, bias_name, lr_mul, bias_init)
    return module
