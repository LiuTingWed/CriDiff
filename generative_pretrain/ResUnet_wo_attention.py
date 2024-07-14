# Realtive import
import sys
sys.path.append('../blocks')
import torch.nn.functional as F
import torch
from torch import nn
# from module.ResNet import ConvNextBlock
import math
from inspect import isfunction
# from torch.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, context=None, *args, **kwargs):
        return self.fn(x, context, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb,self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

from einops.layers.torch import Rearrange
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm(x)
        return self.fn(x,context)



# helpers functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.ModuleList):
            for j, k in m.named_children():
                for a, s in k.named_children():
                    if isinstance(s, nn.Conv2d):
                        nn.init.kaiming_normal_(s.weight, mode='fan_in', nonlinearity='relu')
                        if s.bias is not None:
                            nn.init.zeros_(s.bias)
                    elif isinstance(s, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                        nn.init.ones_(s.weight)
                        if s.bias is not None:
                            nn.init.zeros_(s.bias)

        else:
            pass

import copy

from einops import rearrange, reduce,repeat
from torch import nn, einsum


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class LinearCorssAttention(nn.Module):
    def __init__(self, dim, context_in=None, heads = 4, dim_head = 32):
        super(LinearCorssAttention, self).__init__()

        context_in = default(context_in, dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv2d(context_in, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv2d(context_in, hidden_dim, 1, bias = False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # b, c, h, w = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), (q, k, v))
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class ConvBlock(nn.Module):
    def  __init__(self, dim, dim_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)
        return h


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        self_condition = True,
        with_time_emb = True,
        residual = False
    ):
        super(Unet, self).__init__()

        # self.condition_extractor = ConditionExtractor(dim, dim_mults, False, with_time_emb, residual)

        # determizne dimensions
        input_img_channels = 1
        mask_channels= 1
        side_unit_channel = 64
        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition
        self.side_unit_channel = side_unit_channel


        # init_dim = default(init_dim, dim)

        # self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)

        dim = 64
        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)
        self.init_conv = nn.Conv2d(mask_channels, dim, 7, padding = 3)
        # self.init_conv_cond = nn.Conv2d(input_img_channels, dim, 7, padding = 3)

        self.channels = self.input_img_channels
        self.residual = residual
        # dims_rgb = [dim, *map(lambda m: dim * m, dim_mults)]
        dims_mask = [dim, *map(lambda m: dim * m, dim_mults)]

        # in_out_rgb = list(zip(dims_rgb[:-1], dims_rgb[1:]))
        in_out_mask = list(zip(dims_mask[:-1], dims_mask[1:]))
        from functools import partial
        block_klass = partial(ResnetBlock, groups = 8)


        full_self_attn: tuple = (False, False, False, True)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # attention related params

        attn_kwargs = dict(
            dim_head = 32,
            heads = 4
        )


        self.downs_input = nn.ModuleList([])
        self.downs_label_noise = nn.ModuleList([])
        self.side_out_sup = nn.ModuleList([])
        self.body_sup = nn.ModuleList([])
        self.detail_sup = nn.ModuleList([])



        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out_mask)


        # from module.attention import SpatialTransformer
        # for ind, (dim_in, dim_out) in enumerate(in_out_mask):
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out_mask, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            # dim_head = dim_out // num_heads s
            # attn_klass = Attention if full_attn else LinearAttention

            self.downs_label_noise.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                # Residual(PreNorm(dim_in, LinearCorssAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))


        mid_dim = dims_mask[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearCorssAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # self.cond_mid_block1 = block_klass(512, mid_dim, time_emb_dim = time_dim)



        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out_mask), reversed(full_self_attn))):

            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_in*3, dim_in, time_emb_dim = time_dim) if ind < 3 else block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                # Residual(PreNorm(dim_in, LinearCorssAttention(dim_in))),
                Upsample(dim_in, dim_in) if not is_last else  nn.Conv2d(dim_in, dim_in, 3, padding = 1)
            ]))

        # out_dim = default(mask_channels, input_img_channels)
        self.final_res_block = block_klass(dim, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Sequential(
            # block_klass(dim, output_channels, time_emb_dim = time_dim),
            nn.Conv2d(dim, output_channels, 1),
        )


    def normalization(channels):
        """
        Make a standard normalization layer.

        :param channels: number of input channels.
        :return: an nn.Module for normalization.
        """
        return GroupNorm32(32, channels)



    # def forward(self, input):
    def forward(self, input, time, x_self_cond):

        B,C,H,W, = input.shape
        device = input.device
        x = input
        if self.self_condition:
            tens = torch.zeros((B,1,H,W)).to(device)
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(tens))
            x = torch.cat((x, x_self_cond), dim=1)
        if self.residual:
            orig_x = input

        x = self.init_conv(x)
        # cond = self.init_conv_cond(input)
        # r = x.clone()

        # label = label_noise_t
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        label_noise_side = []
        num = 0
        # out_body_detail = out_body_detail[::-1]
        # out_body = out_body[::-1]
        # out_detail = out_detail[::-1]
        for convnext, convnext2, downsample in self.downs_label_noise:
            x = convnext(x, t)
            label_noise_side.append(x)
            x = convnext2(x, t)
            label_noise_side.append(x)
            x = downsample(x)
            num = num + 1

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        num = 0
        for convnext, convnext2, upsample in self.ups:

            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = torch.cat((x, label_noise_side.pop()), dim=1)

            x = convnext2(x, t)
            x = upsample(x)
            num = num + 1

        if self.residual:
            return self.final_conv(x)
        # x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = Unet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=False
    ).cuda()
    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R, time, None)