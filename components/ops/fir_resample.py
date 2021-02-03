# FIR resample layer.
# Reference and adapted from:
#  https://github.com/NVlabs/stylegan2
#  https://github.com/rosinality/stylegan2-pytorch
# Author: Xiao Li

import os

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

import numpy as np
from typing import List, Union

from torch.cuda.amp import custom_fwd, custom_bwd

module_path = os.path.dirname(__file__)
build_path = os.path.join(module_path, 'build')
os.makedirs(build_path, exist_ok=True)

upfirdn2d_op = load(
    'upfirdn2d',
    sources=[
        os.path.join(module_path, 'cuda_kernel', 'upfirdn2d.cpp'),
        os.path.join(module_path, 'cuda_kernel', 'upfirdn2d_kernel.cu'),
    ],
    verbose=True,
    build_directory=build_path
)


def makeKernel(k: List, nDim: int):
    r"""
    Convert 1D impulse response to normalized N-D conv kernel weight.
    The strict_conv option controls whether the weights should be flipped,
    so that the operation strictly conducts "mathmatical" convlution instead
    of cross-corrlation.
    """
    # Out product nDim times
    k_weight1D = np.array(k)
    k_weight = np.array(k)
    for _k in range(nDim - 1):
        k_weight = np.kron(k_weight, k_weight1D)
        k_weight = k_weight.reshape([len(k)] * (_k + 2))

    k_tensor = torch.tensor(k_weight, dtype=torch.float32)
    k_tensor /= k_tensor.sum()
    return k_tensor


class UpFirDn2dBackward(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    @custom_bwd
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out


# !!!DO NOT REMOVE ABOVE COMMENTS!!!
# A refernce of upfirdn2d in pure pytorch ops. Leave the code here for reference.
# def upfirdn2d_native(input, kernel, up=1, down=1, pad=(0, 0)):
#     _, minor, in_h, in_w = input.shape
#     kernel_h, kernel_w = kernel.shape

#     up_x, up_y = up, up
#     down_x, down_y = down, down
#     pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]

#     out = input.view(-1, in_h, 1, in_w, 1, minor)
#     out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
#     out = out.view(-1, in_h * up_y, in_w * up_x, minor)

#     out = F.pad(
#         out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
#     )
#     out = out[
#         :,
#         max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
#         max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
#         :,
#     ]

#     out = out.permute(0, 3, 1, 2)
#     out = out.reshape(
#         [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
#     )
#     w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
#     out = F.conv2d(out, w)
#     out = out.reshape(
#         -1,
#         minor,
#         in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
#         in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
#     )
#     out = out[:, :, ::down_y, ::down_x]

#     return out


class FIRResample2d(nn.Module):
    r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

    Basically it performs the following operations for each image:
    1. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
       corresponds to cropping the image.
    2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).
    3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.

       Some common binomial pattern example: [1, 1] (Nearest interpolation),
       [1,2,1] (Linear interpolation, StyleGANv1 default),
       [1,3,3,1] (StyleGANv2 default),
       [1,4,6,4,1].

    4. Downsample the image by throwing away pixels (`downx`, `downy`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation using
    Pure pytorch ops.
    """

    def __init__(self, up_factor: int, down_factor: int, fir_kernel: List,
                 gain: float = 1.0, pad: Union[List[int], int] = 0):

        super(FIRResample2d, self).__init__()

        fir_kernel = makeKernel(fir_kernel, 2) * gain
        self.register_buffer('fir_kernel', fir_kernel)

        if(isinstance(pad, int)):
            pad = (pad, pad)
        self.pad = pad

        self.up_factor = up_factor
        self.down_factor = down_factor

    def forward(self, x: torch.Tensor):
        out = upfirdn2d(x, self.fir_kernel, up=self.up_factor, down=self.down_factor, pad=self.pad)
        return out
