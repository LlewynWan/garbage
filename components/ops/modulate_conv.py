# Modulated Conv Layer used in StyleGAN v2.
# Reference and adapted from:
#  https://github.com/NVlabs/stylegan2
#  https://github.com/facebookresearch/pytorch_GAN_zoo
#  https://github.com/rosinality/stylegan2-pytorch
# Author: Xiao Li

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .equalized_layer import EqualizeLearningRate


def modulate_and_demodulate_weight(weight: torch.Tensor, style: torch.Tensor,
                                   demodulate: bool = True,
                                   transposed_weight: bool = False):
    r"""
    Perform weight modulation and demodulation as described in StyleGAN v2.
    Please refer to section 2.2 as well as Appendix B of StyleGAN v2 paper for technical details.

    Note: For modulate weight from ConvTranspose layers, set transposed_weight to True.
    """
    b, ch_in = style.shape
    n_dim_spatial = len(weight.shape) - 2

    # Modulate
    if(transposed_weight):
        _style_view_shape = [b, ch_in, 1] + [1] * n_dim_spatial
    else:
        _style_view_shape = [b, 1, ch_in] + [1] * n_dim_spatial
    weight_out = weight.unsqueeze(0) * style.view(*_style_view_shape)

    if(demodulate):
        # Demodulate
        if(transposed_weight):
            demod_dims = [1] + list(range(3, n_dim_spatial + 3))
        else:
            demod_dims = [2] + list(range(3, n_dim_spatial + 3))
        demod_factor = torch.rsqrt(torch.square(weight_out).sum(demod_dims, keepdim=True) + 1e-8)
        weight_out = weight_out * demod_factor

    return weight_out


class ModulatedConv2d(nn.Conv2d):
    r"""
    Modulated Conv2d layer in StyleGANv2.
    demod: perform demodulate or not.

    Note:
        - Modulated Conv / ConvTranspose do not use bias parameter.
        - Modulated Conv / ConvTranspose currently do not support
          group convolution because internally it utilizes group convlution
          to do the computational trick.
          Please see the StyleGAN v2 paper for details.
    """
    def __init__(self, *args, demod: bool = True, **kwargs):
        # Force bias to false.
        kwargs['bias'] = False
        super(ModulatedConv2d, self).__init__(*args, **kwargs)

        self.demod = demod

    def forward(self, feat: torch.Tensor, style: torch.Tensor):
        b, ch_in, height, width = feat.shape

        # Mod/Demod.
        conv_weight = modulate_and_demodulate_weight(self.weight, style, demodulate=self.demod)
        # Re-grouping
        conv_weight = conv_weight.view(
            b * self.out_channels, ch_in, *self.kernel_size
        )
        # Re-grouping input
        feat = feat.view(1, b * ch_in, height, width)

        # Difference from original nn.Conv2d call:
        # (1) weight is modulated
        # (2) group is set to batchsize for computing with modulated weights
        # (3) bias is always None. ModulatedConv do not use bias.
        if self.padding_mode != 'zeros':
            out = F.conv2d(F.pad(feat, self._padding_repeated_twice,
                           mode=self.padding_mode),
                           conv_weight, None, self.stride,
                           _pair(0), self.dilation, b)
        else:
            out = F.conv2d(feat, conv_weight, None, self.stride,
                           self.padding, self.dilation, b)

        # Group back to original shape.
        _, _, height, width = out.shape
        out = out.view(b, self.out_channels, height, width)

        return out


class ModulatedConvTranspose2d(nn.ConvTranspose2d):
    r"""
    Modulated Conv2d layer in StyleGANv2.
    demod: perform demodulate or not.

    Note:
        - Modulated Conv / ConvTranspose do not use bias parameter.
        - Modulated Conv / ConvTranspose currently do not support
          group convolution because internally it utilizes group convlution
          to do the computational trick.
          Please see the StyleGAN v2 paper for details.
    """
    def __init__(self, *args, demod: bool = True, **kwargs):
        # Force bias to false.
        kwargs['bias'] = False
        super(ModulatedConvTranspose2d, self).__init__(*args, **kwargs)

        self.demod = demod

    def forward(self, feat: torch.Tensor, style: torch.Tensor, output_size=None):
        b, ch_in, height, width = feat.shape

        # Mod/Demod.
        conv_weight = modulate_and_demodulate_weight(self.weight, style,
                                                     demodulate=self.demod, transposed_weight=True)
        # Re-grouping weight
        conv_weight = conv_weight.view(
            b * ch_in, self.out_channels, *self.kernel_size
        )
        # Re-grouping input
        feat = feat.view(1, b * ch_in, height, width)

        # Difference from original nn.Conv2d call:
        # (1) weight is modulated
        # (2) group is set to batchsize for computing with modulated weights
        # (3) bias is always None. ModulatedConv do not use bias.
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(feat,
                                              output_size,
                                              self.stride,
                                              self.padding,
                                              self.kernel_size)
        out = F.conv_transpose2d(feat, conv_weight, None,
                                 self.stride, self.padding,
                                 output_padding, b, self.dilation)

        # Group back to original shape.
        _, _, height, width = out.shape
        out = out.view(b, self.out_channels, height, width)

        return out
