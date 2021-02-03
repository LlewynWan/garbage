# Fused bias and LeakyReLU activate function.
# Reference:
#  https://github.com/NVlabs/stylegan2
#  https://github.com/rosinality/stylegan2-pytorch

import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

from torch.cuda.amp import custom_fwd, custom_bwd

module_path = os.path.dirname(__file__)
build_path = os.path.join(module_path, 'build')
os.makedirs(build_path, exist_ok=True)

fused = load(
    'fused',
    sources=[
        os.path.join(module_path, 'cuda_kernel', 'fused_bias_act.cpp'),
        os.path.join(module_path, 'cuda_kernel', 'fused_bias_act_kernel.cu'),
    ],
    verbose=True,
    build_directory=build_path
)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    @custom_bwd
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, bias_scale=1.0, act_scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.bias_scale = bias_scale
        self.act_scale = act_scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias * self.bias_scale, self.negative_slope, self.act_scale)


class FusedLeakyReLUPython(nn.Module):
    def __init__(self, channel, negative_slope=0.2, bias_scale=1.0, act_scale=2 ** 0.5):
        super(FusedLeakyReLUPython, self).__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.bias_scale = bias_scale
        self.act_scale = act_scale

    def forward(self, input):
        bias_shape = [1, input.shape[1]] + list([1] * len(input.shape[2::]))
        output = input + self.bias_scale * self.bias.view(*bias_shape)
        output = self.act_scale * F.leaky_relu(output, self.negative_slope)

        return output
