from typing import Optional, List, Union

import torch
import torch.nn as nn

import math

from .ops import EqualizeLearningRate
from .ops import FusedLeakyReLU
from .ops import ModulatedConv2d, ModulatedConvTranspose2d, FIRResample2d


def get_ch_at_res(res: int, ch_base: int, ch_min: int, ch_max: int):
    return min(max(ch_base // res, ch_min), ch_max)


class LinearWithLeakyReLU(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, lr_mul: float = 0.01):
        super(LinearWithLeakyReLU, self).__init__()

        linear_layer = EqualizeLearningRate(
                nn.Linear(in_channel, out_channel, bias=False),
                'weight', None,
                lr_mul=lr_mul, bias_init=0.0)
        bias_act_layer = FusedLeakyReLU(out_channel, bias_scale=lr_mul, act_scale=math.sqrt(2))

        self.layer = nn.Sequential(
                linear_layer,
                bias_act_layer)

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class G_Synthesis_BaseOps:
    class NoiseInjection(nn.Module):
        def __init__(self):
            super(G_Synthesis_BaseOps.NoiseInjection, self).__init__()
            self.noise_scale = nn.Parameter(torch.zeros(1))

        def forward(self, feat: torch.Tensor, override_noise: Optional[torch.Tensor] = None):
            if override_noise is None:
                batch, _, height, width = feat.shape
                override_noise = feat.new_empty(batch, 1, height, width).normal_()

            return feat + self.noise_scale * override_noise

    class BottleneckModConv(nn.Module):
        def __init__(self, in_channel: int, out_channel: int, kernel_size: int, inject_noise: bool = False):
            super(G_Synthesis_BaseOps.BottleneckModConv, self).__init__()

            self.conv_layer = EqualizeLearningRate(
                    ModulatedConv2d(
                        in_channel, out_channel, kernel_size,
                        stride=1, padding = kernel_size//2, bias=False
                    ),
                    'weight', None)
            self.noise_inject_layer = G_Synthesis_BaseOps.NoiseInjection()
            self.act_layer = FusedLeakyReLU(out_channel, act_scale=math.sqrt(2))
            self.inject_noise = inject_noise

        def forward(self, feat: torch.Tensor, style: torch.Tensor,
                override_noise: Optional[torch.Tensor] = None):
            out = self.conv_layer(feat, style)
            if self.inject_noise:
                out = self.noise_inject_layer(out, override_noise)
            out = self.act_layer(out)

            return out

    class Upsample2xModConv(nn.Module):
        def __init__(self,
                in_channel: int, out_channel: int, kernel_size: int,
                inject_noise: bool = False,
                blur_kernel: List = [1, 3, 3, 1]):

            super(G_Synthesis_BaseOps.Upsample2xModConv, self).__init__()
            self.conv_layer = EqualizeLearningRate(
                    ModulatedConvTranspose2d(
                        in_channel, out_channel, kernel_size,
                        stride=2, padding=0, bias=False,
                    ),
                    'weight', None)

            p = len(blur_kernel) - kernel_size + 1
            p_left = (p + 1) // 2
            p_right = p // 2
            self.fir_layer = FIRResample2d(up_factor=1, down_factor=1, fir_kernel=blur_kernel, gain=4.0, pad=[p_left, p_right])
            self.act_layer = FusedLeakyReLU(out_channel, act_scale=math.sqrt(2))
            self.noise_inject_layer = G_Synthesis_BaseOps.NoiseInjection()
            self.inject_noise = inject_noise

        def forward(self, feat: torch.Tensor, style: torch.Tensor,
                override_noise: Optional[torch.Tensor] = None):
            out = self.conv_layer(feat, style)
            out = self.fir_layer(out)
            if self.inject_noise:
                out = self.noise_inject_layer(out, override_noise)
            out = self.act_layer(out)

            return out

    class ToOutput(nn.Module):
        def __init__(self,
                in_channel: int,
                out_channel: int):
            super(G_Synthesis_BaseOps.ToOutput, self).__init__()

            self.conv_layer = EqualizeLearningRate(
                    ModulatedConv2d(
                        in_channel, out_channel, 1,
                        stride=1, padding=0, bias=False,
                        demod=False
                    ),
                    'weight', None)
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

        def forward(self, feat: torch.Tensor, style: torch.Tensor, prev_output: Optional[torch.Tensor] = None):
            out = self.conv_layer(feat, style)
            out = out + self.bias
            if prev_output is not None:
                out = out + prev_output

            return out


class G_Synthesis_ResNet(nn.Module):
    class Upsample2xConv(nn.Module):
        def __init__(self,
                in_channel: int, out_channel: int, kernel_size: int,
                blur_kernel: List = [1, 3, 3, 1],
                act: bool = True):
            
            super(G_Synthesis_ResNet.Upsample2xConv, self).__init__()
            conv_layer = EqualizeLearningRate(
                    nn.ConvTranspose2d(
                        in_channel, out_channel, kernel_size,
                        stride=2, padding=0, bias=False
                        ),
                    'weight', None)

            p = len(blur_kernel) - kernel_size + 1
            p_left = (p + 1) // 2
            p_right = p // 2
            fir_layer = FIRResample2d(up_factor=1, down_factor=1, fir_kernel=blur_kernel, gain=4.0, pad=[p_left, p_right])

            layer_list = [conv_layer, fir_layer]
            if act:
                layer_list.append(FusedLeakyReLU(out_channel, act_scale=math.sqrt(2)))

            self.layers = nn.Sequential(*layer_list)

        def forward(self, feat: torch.Tensor):
            return self.layers(feat)
    
    class FirstSynthesisBlock(nn.Module):
        def __init__(self, in_channel: int, btk_channel: int, out_channel: int,
                kernel_size: int, inject_noise: bool = False, blur_kernel: List = [1, 3, 3, 1]):
            super(G_Synthesis_ResNet.FirstSynthesisBlock, self).__init__()

            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(in_channel,
                    btk_channel, kernel_size, inject_noise)

        def forward(self, init_feat: torch.Tensor, style: torch.Tensor, override_noise: Optional[torch.Tensor] = None):
            next_feat = self.conv_btk(init_feat, style, override_noise)
            return next_feat

    class ResBottleneckSynthesisBlock(nn.Module):
        def __init__(self, in_channel: int, out_channel: int,
                kernel_size: int, inject_noise: List[bool], blur_kernel: List = [1, 3, 3, 1]):
            super(G_Synthesis_ResNet.ResBottleneckSynthesisBlock, self).__init__()

            self.conv_0 = G_Synthesis_ResNet.Upsample2xConv(in_channel, in_channel // 2, 1, blur_kernel)
            self.conv_1 = G_Synthesis_BaseOps.BottleneckModConv(in_channel // 2, in_channel // 2, kernel_size, inject_noise[0])
            self.conv_2 = G_Synthesis_BaseOps.BottleneckModConv(in_channel // 2, out_channel, 1, inject_noise[1])

            self.skip = G_Synthesis_ResNet.Upsample2xConv(in_channel, out_channel, 1, blur_kernel, act=False)

        def forward(self, prev_feat: torch.Tensor, style: torch.Tensor, override_noise: List[torch.Tensor]):
            assert(style.shape[0] == 2 and len(override_noise) == 2)

            next_feat = self.conv_0(prev_feat)
            next_feat = self.conv_1(next_feat, style[0], override_noise[0])
            next_feat = self.conv_2(next_feat, style[1], override_noise[1])
            prev_feat_up = self.skip(prev_feat)

            out_feat = (next_feat + prev_feat_up) * (1.0 / math.sqrt(2))

            return out_feat

    class ResSynthesisBlock(nn.Module):
        def __init__(self, in_channel: int, out_channel: int,
                kernel_size: int, inject_noise: List[bool], blur_kernel: List = [1, 3, 3, 1]):
            super(G_Synthesis_ResNet.ResSynthesisBlock, self).__init__()
            
            self.conv_up = G_Synthesis_BaseOps.Upsample2xModConv(in_channel, out_channel, kernel_size, inject_noise[0], blur_kernel)
            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(out_channel, out_channel, kernel_size, inject_noise[1])

            self.upsample = G_Synthesis_ResNet.Upsample2xConv(in_channel, out_channel, 1, blur_kernel, act=False)

        def forward(self, prev_feat: torch.Tensor, style: torch.Tensor, override_noise: List[torch.Tensor]):
            assert(style.shape[0] == 2 and len(override_noise) == 2)

            next_feat = self.conv_up(prev_feat, style[0], override_noise[0])
            next_feat = self.conv_btk(next_feat, style[1], override_noise[1])
            prev_feat_up = self.upsample(prev_feat)

            out_feat = (next_feat + prev_feat_up) * (1.0 / math.sqrt(2))

            return out_feat

    def __init__(self,
            feat_res_min: int = 4,
            feat_res_max: int = 256,
            ch_res1_unclip: int = 8192,
            ch_max: int = 512,
            ch_min: int = 1,
            ch_out: int = 3,
            blur_kernel: List = [1, 3, 3, 1],
            inject_noise: Optional[Union[List[bool], bool]] = False):

        super(G_Synthesis_ResNet, self).__init__()
        
        self.n_res = int(math.log2(feat_res_max // feat_res_min)) + 1
        self.n_w = self.n_res * 2
        self.n_noise = self.n_w - 1

        if isinstance(inject_noise, bool):
            inject_noise = [inject_noise] * self.n_noise

        self.feat_res_min = feat_res_min
        self.feat_res_max = feat_res_max
        self.ch_res1_unclip = ch_res1_unclip
        self.ch_max = ch_max
        self.ch_min = ch_min
        self.ch_out = ch_out

        self.first_block = G_Synthesis_ResNet.FirstSynthesisBlock(
            get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max),
            get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max), ch_out,
            3, inject_noise[0], blur_kernel
        )

        self.layers = nn.ModuleDict()
        id_cursor = 1
        res_cursor = feat_res_min * 2
        while res_cursor <= feat_res_max:
            self.layers['block_{}-{}'.format(res_cursor // 2, res_cursor)] = G_Synthesis_ResNet.ResSynthesisBlock(
                get_ch_at_res(res_cursor // 2, ch_res1_unclip, ch_min, ch_max),
                get_ch_at_res(res_cursor, ch_res1_unclip, ch_min, ch_max),
                3, inject_noise[id_cursor:id_cursor + 2], blur_kernel
            )
            res_cursor *= 2
            id_cursor += 2

        self.to_output = G_Synthesis_BaseOps.ToOutput(get_ch_at_res(res_cursor // 2, ch_res1_unclip, ch_min, ch_max), ch_out)

    def forward(self, init_feat: torch.Tensor, style: torch.Tensor, override_noise: Optional[List[torch.Tensor]] = None):
        if(override_noise is None):
            override_noise = [None] * self.n_noise

        assert(len(override_noise) == self.n_noise and style.shape[0] == self.n_w)

        out_feat = self.first_block(init_feat, style[0], override_noise[0])

        for j, key in enumerate(self.layers):
            i = 2 * j + 1
            layer = self.layers[key]
            out_feat = layer(out_feat, style[i:i + 2], override_noise[i:i + 2])

        out = self.to_output(out_feat, style[-1])
        return out


class G_Synthesis_Skip(nn.Module):
    class Upsample2x(FIRResample2d):
        def __init__(self, blur_kernel: List):
            p = len(blur_kernel) - 1
            p_left = (p + 1) // 2
            p_right = p // 2

            super(G_Synthesis_Skip.Upsample2x, self).__init__(up_factor=2, down_factor=1, fir_kernel=blur_kernel, gain=4.0, pad=[p_left, p_right])

    class FirstSynthesisBlock(nn.Module):
        def __init__(self, in_channel: int, btk_channel: int, out_channel: int,
                kernel_size: int, inject_noise: bool = False, blur_kernel: List = [1, 3, 3, 1]):
            super(G_Synthesis_Skip.FirstSynthesisBlock, self).__init__()

            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(in_channel, btk_channel, kernel_size, inject_noise)
            self.to_output = G_Synthesis_BaseOps.ToOutput(btk_channel, out_channel)

        def forward(self, init_feat: torch.Tensor, style: torch.Tensor, override_noise: Optional[torch.Tensor] = None):
            assert(style.shape[0] == 2)

            next_feat = self.conv_btk(init_feat, style[0], override_noise)
            next_output = self.to_output(next_feat, style[1])

            return next_feat, next_output

    class SkipSynthesisBlock(nn.Module):
        def __init__(self, in_channel: int, btk_channel: int, out_channel: int,
                kernel_size: int, inject_noise: List[bool], blur_kernel: List = [1, 3, 3, 1]):
            super(G_Synthesis_Skip.SkipSynthesisBlock, self).__init__()

            self.conv_up = G_Synthesis_BaseOps.Upsample2xModConv(in_channel, btk_channel, kernel_size, inject_noise[0], blur_kernel)
            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(btk_channel, btk_channel, kernel_size, inject_noise[1])
            self.to_output = G_Synthesis_BaseOps.ToOutput(btk_channel, out_channel)
            self.fir_up = G_Synthesis_Skip.Upsample2x(blur_kernel)

        def forward(self, prev_feat: torch.Tensor, style: torch.Tensor,
                    prev_output: torch.Tensor, override_noise: List[torch.Tensor]):
            assert(style.shape[0] == 3 and len(override_noise) == 2)

            next_feat = self.conv_up(prev_feat, style[0], override_noise[0])
            next_feat = self.conv_btk(next_feat, style[1], override_noise[1])

            prev_output_up = self.fir_up(prev_output)
            next_output = self.to_output(next_feat, style[2], prev_output_up)

            return next_feat, next_output

    def __init__(self,
                 feat_res_min: int = 4,
                 feat_res_max: int = 256,
                 ch_res1_unclip: int = 8192,
                 ch_max: int = 512,
                 ch_min: int = 1,
                 ch_out: int = 3,
                 blur_kernel: List = [1, 3, 3, 1],
                 inject_noise: Optional[Union[List[bool], bool]] = False):

        super(G_Synthesis_Skip, self).__init__()

        self.n_res = int(math.log2(feat_res_max // feat_res_min)) + 1
        self.n_w = self.n_res * 2
        self.n_noise = self.n_w - 1

        if isinstance(inject_noise, bool):
            inject_noise = [inject_noise] * self.n_noise

        self.feat_res_min = feat_res_min
        self.feat_res_max = feat_res_max
        self.ch_res1_unclip = ch_res1_unclip
        self.ch_max = ch_max
        self.ch_min = ch_min
        self.ch_out = ch_out

        self.first_block = G_Synthesis_Skip.FirstSynthesisBlock(
            get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max),
            get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max), ch_out,
            3, inject_noise[0], blur_kernel)

        self.layers = nn.ModuleDict()
        id_cursor = 1
        res_cursor = feat_res_min * 2
        while res_cursor <= feat_res_max:
            self.layers['block_{}-{}'.format(res_cursor // 2, res_cursor)] = G_Synthesis_Skip.SkipSynthesisBlock(
                get_ch_at_res(res_cursor // 2, ch_res1_unclip, ch_min, ch_max),
                get_ch_at_res(res_cursor, ch_res1_unclip, ch_min, ch_max), ch_out,
                3, inject_noise[id_cursor:id_cursor + 2], blur_kernel
            )
            res_cursor *= 2
            id_cursor += 2

    def forward(self, init_feat: torch.Tensor, style: torch.Tensor, override_noise: Optional[List[torch.Tensor]] = None):
        if(override_noise is None):
            override_noise = [None] * self.n_noise

        assert(len(override_noise) == self.n_noise and style.shape[0] == self.n_w)

        out_feat, out = self.first_block(init_feat, style[0:2], override_noise[0])

        for j, key in enumerate(self.layers):
            i = 2 * j + 1
            layer = self.layers[key]
            out_feat, out = layer(out_feat, style[i:i + 3], out, override_noise[i:i + 2])

        return out


class StyleEncoder(nn.Module):
    class Downsample2x(FIRResample2d):
        def __init__(self, blur_kernel: List):
            p = len(blur_kernel) - 1
            p_left = (p + 1) // 2
            p_right = p // 2

            super(StyleEncoder.Downsample2x, self).__init__(up_factor=1, down_factor=2, fir_kernel=blur_kernel, gain=4.0, pad=[p_left, p_right])

    class Downsample2xModConv(nn.Module):
        def __init__(self,
                in_channel: int, out_channel: int, kernel_size: int,
                inject_noise: bool = False,
                blur_kernel: List = [1, 3, 3, 1]):

            super(StyleEncoder.Downsample2xModConv, self).__init__()
            self.conv_layer = EqualizeLearningRate(
                    ModulatedConv2d(
                        in_channel, out_channel, kernel_size,
                        stride=2, padding=0, bias=False
                    ),
                    'weight', None)

            p = len(blur_kernel) + kernel_size - 3
            p_left = (p + 1) // 2
            p_right = p // 2

            self.fir_layer = FIRResample2d(up_factor=1, down_factor=1, fir_kernel=blur_kernel, gain=1.0, pad=[p_left, p_right])
            self.act_layer = FusedLeakyReLU(out_channel, act_scale=math.sqrt(2))
            self.noise_inject_layer = G_Synthesis_BaseOps.NoiseInjection()
            self.inject_noise = inject_noise

        def forward(self, feat: torch.Tensor, style: torch.Tensor, override_noise: Optional[List[torch.Tensor]]):
            out = self.fir_layer(feat)
            out = self.conv_layer(out, style)
            if self.inject_noise:
                out = self.noise_inject_layer(out, override_noise)
            out = self.act_layer(out)

            return out

    class FirstEncodingBlock(nn.Module):
        def __init__(self, in_channel: int, btk_channel: int, out_channel: int,
                kernel_size: int, inject_noise: bool = False, blur_kernel: List = [1, 3, 3, 1]):
            super(StyleEncoder.FirstEncodingBlock, self).__init__()

            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(in_channel, btk_channel, kernel_size, inject_noise)
            self.to_output = G_Synthesis_BaseOps.ToOutput(btk_channel, out_channel)

        def forward(self, frame: torch.Tensor, style: torch.Tensor, override_noise: Optional[torch.Tensor] = None):
            assert(style.shape[0] == 2)

            next_feat = self.conv_btk(frame, style[0], override_noise)
            next_output = self.to_output(next_feat, style[1])

            return next_feat, next_output

    class SkipEncodingBlock(nn.Module):
        def __init__(self, in_channel: int, btk_channel: int, out_channel: int,
                kernel_size: int, inject_noise: List[bool], blur_kernel: List = [1, 3, 3, 1]):
            super(StyleEncoder.SkipEncodingBlock, self).__init__()

            self.conv_down = StyleEncoder.Downsample2xModConv(in_channel, btk_channel, kernel_size, inject_noise[0], blur_kernel)
            self.conv_btk = G_Synthesis_BaseOps.BottleneckModConv(btk_channel, btk_channel, kernel_size, inject_noise[1])
            self.to_output = G_Synthesis_BaseOps.ToOutput(btk_channel, out_channel)
            self.fir_down = StyleEncoder.Downsample2x(blur_kernel)

        def forward(self, prev_feat: torch.Tensor, style: torch.Tensor,
                prev_output: torch.Tensor, override_noise: List[torch.Tensor]):
            assert(style.shape[0] == 3 and len(override_noise) == 2)

            next_feat = self.conv_down(prev_feat, style[0], override_noise[0])
            next_feat = self.conv_btk(next_feat, style[1], override_noise[1])

            prev_output_down = self.fir_down(prev_output)
            next_output = self.to_output(next_feat, style[2], prev_output_down)

            return next_feat, next_output

    def __init__(self,
            feat_res_min: int = 4,
            feat_res_max: int = 256,
            ch_res1_unclip: int = 8192,
            ch_max: int = 512,
            ch_min: int = 1,
            ch_in: int = 3,
            blur_kernel: List = [1, 3, 3, 1],
            inject_noise: Optional[Union[List[bool], bool]] = False):

        super(StyleEncoder, self).__init__()
        
        self.n_res = int(math.log2(feat_res_max // feat_res_min)) + 1
        self.n_w = self.n_res * 2
        self.n_noise = self.n_w - 1

        if isinstance(inject_noise, bool):
            inject_noise = [inject_noise] * self.n_noise

        self.feat_res_min = feat_res_min
        self.feat_res_max = feat_res_max
        self.ch_res1_unclip = ch_res1_unclip
        self.ch_max = ch_max
        self.ch_min = ch_min
        self.ch_in = ch_in

        self.layers = nn.ModuleDict()
        self.first_block = StyleEncoder.FirstEncodingBlock(ch_in,
                get_ch_at_res(feat_res_max, ch_res1_unclip, ch_min, ch_max),
                get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max),
                3, inject_noise[0], blur_kernel)

        id_cursor = 1
        res_cursor = feat_res_max // 2
        while (res_cursor >= feat_res_min):
            self.layers['block_{}-{}'.format(res_cursor * 2, res_cursor)] = StyleEncoder.SkipEncodingBlock(
                get_ch_at_res(res_cursor * 2, ch_res1_unclip, ch_min, ch_max),
                get_ch_at_res(res_cursor, ch_res1_unclip, ch_min, ch_max),
                get_ch_at_res(feat_res_min, ch_res1_unclip, ch_min, ch_max),
                3, inject_noise[id_cursor:id_cursor + 2], blur_kernel)
            
            res_cursor = res_cursor // 2
            id_cursor += 2

    def forward(self, frame: torch.Tensor,
            style: Union[List[torch.Tensor], torch.Tensor],
            override_noise: Optional[Union[List[torch.Tensor], torch.Tensor]] = None):
        if(override_noise is None):
            override_noise = [None] * self.n_noise

        assert(len(override_noise) == self.n_noise and style.shape[0] == self.n_w)

        out_feat, out = self.first_block(frame, style[0:2], override_noise[0])

        for j, key in enumerate(self.layers):
            i = 2 * j + 1
            layer = self.layers[key]
            out_feat, out = layer(out_feat, style[i:i + 3], out, override_noise[i:i + 2])

        return out

class StyleDecoder(nn.Module):
    def __init__(self,
            synthesis_network: str = 'Skip',
            feat_res_min: int = 4,
            feat_res_max: int = 256,
            ch_res1_unclip: int = 8192,
            ch_max: int = 512,
            ch_min: int = 1,
            ch_out: int = 3,
            blur_kernel: List = [1, 3, 3, 1],
            inject_noise: Optional[Union[List[bool], bool]] = False):

        super(StyleDecoder, self).__init__()

        if synthesis_network == 'Skip':
            self.g_synthesis = G_Synthesis_Skip(feat_res_min, feat_res_max,
                    ch_res1_unclip, ch_max, ch_min, ch_out,
                    blur_kernel, inject_noise)
        elif synthesis_network == 'ResNet':
            self.g_synthesis = G_Synthesis_ResNet(feat_res_min, feat_res_max,
                    ch_res1_unclip, ch_max, ch_min, ch_out,
                    blur_kernel, inject_noise)

        self.n_w = self.g_synthesis.n_w

    def forward(self, init_feat: torch.Tensor,
            style: Union[List[torch.Tensor], torch.Tensor],
            override_noise: Optional[Union[List[torch.Tensor], torch.Tensor]] = None):
        fake_img = self.g_synthesis(init_feat, style, override_noise)
        
        return fake_img

class LatentEncoder(nn.Module):
    def __init__(self,
            ch_min: int = 1,
            ch_max: int = 512,
            feat_res_min: int = 4,
            feat_res_max: int = 256,
            ch_res1_unclip: int = 8192):
        pass
