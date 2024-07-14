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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

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
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        #
        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)
        #
        # # attention, what we cannot get enough of
        # attn = sim.softmax(dim=-1)
        # out = torch.einsum('b h d e, b h d n -> b h e n', attn, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class BodyEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(BodyEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c + out_c, out_c)

        # 2 column
        self.col23 = ConvBlock(out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c + out_c, out_c)

        # 3 column
        self.col33 = ConvBlock(out_c, out_c)
        self.col32 = ConvBlock(out_c + out_c + out_c, out_c)

        # 4 column
        self.col43 = ConvBlock(out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage40 = self.col13(stage4)
        stage41 = self.col23(stage40)
        stage42 = self.col33(stage41)
        stage43 = self.col43(stage42)
        side_out.append(stage43)
        stage40 = F.interpolate(stage40, size=(self.iS1,self.iS1), mode='bilinear')
        stage41 = F.interpolate(stage41, size=(self.iS1,self.iS1), mode='bilinear')
        stage42 = F.interpolate(stage42, size=(self.iS1,self.iS1), mode='bilinear')
        stage43 = F.interpolate(stage43, size=(self.iS1,self.iS1), mode='bilinear')

        stage31 = self.col12(torch.cat((stage40, xs[-2]), dim=1))
        stage32 = self.col21(torch.cat((stage41, stage31), dim=1))
        stage33 = self.col32(torch.cat((stage42, stage43, stage32), dim=1))
        side_out.append(stage33)
        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage33 = F.interpolate(stage33, size=(self.iS2,self.iS2), mode='bilinear')

        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col20(torch.cat((stage32, stage33, stage21), dim=1))
        side_out.append(stage22)
        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')

        stage1 = self.col10(torch.cat((stage21, stage22, xs[-4]), dim=1))
        side_out.append(stage1)

        return stage1, side_out

class DetailEnhance(nn.Module):
    def __init__(self, iC_list, iSize_list, out_c):
        super(DetailEnhance, self).__init__()
        ic0, ic1, ic2, ic3 = iC_list
        self.iS0, self.iS1, self.iS2, self.iS3 = iSize_list
        # 1 column
        self.col13 = ConvBlock(out_c, out_c)
        self.col12 = ConvBlock(out_c + out_c, out_c)
        self.col11 = ConvBlock(out_c + out_c, out_c)
        self.col10 = ConvBlock(out_c + out_c, out_c)

        # 2 column
        self.col22 = ConvBlock(out_c + out_c, out_c)
        self.col21 = ConvBlock(out_c + out_c, out_c)
        self.col20 = ConvBlock(out_c + out_c, out_c)

        # 3 column
        self.col31 = ConvBlock(out_c + out_c, out_c)
        self.col30 = ConvBlock(out_c + out_c, out_c)

        # 4 column
        self.col40 = ConvBlock(out_c + out_c, out_c)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        side_out = []
        stage4 = xs[-1]
        stage4 = self.col13(stage4)
        side_out.append(stage4)
        stage4 = F.interpolate(stage4, size=(self.iS1,self.iS1), mode='bilinear')
        stage31 = self.col12(torch.cat((stage4, xs[-2]), dim=1))
        stage32 = self.col22(torch.cat((stage4, stage31), dim=1))
        side_out.append(stage32)

        stage31 = F.interpolate(stage31, size=(self.iS2,self.iS2), mode='bilinear')
        stage32 = F.interpolate(stage32, size=(self.iS2,self.iS2), mode='bilinear')
        stage21 = self.col11(torch.cat((stage31, xs[-3]), dim=1))
        stage22 = self.col21(torch.cat((stage32, stage21), dim=1))
        stage23 = self.col31(torch.cat((stage32, stage22), dim=1))
        side_out.append(stage23)

        stage21 = F.interpolate(stage21, size=(self.iS3,self.iS3), mode='bilinear')
        stage22 = F.interpolate(stage22, size=(self.iS3,self.iS3), mode='bilinear')
        stage23 = F.interpolate(stage23, size=(self.iS3,self.iS3), mode='bilinear')
        stage11 = self.col10(torch.cat((stage21, xs[-4]), dim=1))
        stage12 = self.col20(torch.cat((stage22, stage11), dim=1))
        stage13 = self.col30(torch.cat((stage23, stage12), dim=1))
        stage14 = self.col40(torch.cat((stage23, stage13), dim=1))
        side_out.append(stage14)

        return stage14, side_out

class Decoder_body_detail(nn.Module):
    def __init__(self,in_c, out_c):
        super(Decoder_body_detail, self).__init__()
        self.conv0 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(out_c)
        self.ConvBlock0 = ConvBlock(out_c, out_c)

        self.conv1 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.ConvBlock1 = ConvBlock(out_c, out_c)

        self.conv2 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.ConvBlock2 = ConvBlock(out_c, out_c)

        self.conv3 = nn.Conv2d(in_c + out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        # self.ConvBlock3 = ConvBlock(64, 32)


    def forward(self, input1, input2):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out0 = self.ConvBlock0(out0)
        out1 = torch.cat(((input1[1] + input2[1]), out0), dim=1)
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out_list.append(out1)

        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out1 = self.ConvBlock1(out1)
        out2 = torch.cat(((input1[2] + input2[2]), out1), dim=1)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out_list.append(out2)

        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out2 = self.ConvBlock2(out2)
        out3 = torch.cat(((input1[3] + input2[3]), out2), dim=1)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out_list.append(out3)
        return out_list


class Decoder_detail(nn.Module):
    def __init__(self):
        super(Decoder_detail, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # self.FEM1 = Frequency_Edge_Module(radius=16, channel=64)
        # self.FEM2 = Frequency_Edge_Module(radius=16, channel=128)


    def forward(self, input1, input2=[0, 0, 0, 0]):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(  torch.cat((input1[1],out0), dim=1) )), inplace=True)
        out_list.append(out1)


        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        # input12 = self.FEM2(input1[2])
        out2 = F.relu(self.bn2(self.conv2(   torch.cat((input1[2],out1), dim=1 ))), inplace=True)
        out_list.append(out2)


        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        # input13 = self.FEM1(input1[3])
        out3 = F.relu(self.bn3(self.conv3(  torch.cat((input1[3],out2), dim=1 ))), inplace=True)
        out_list.append(out3)
        return out_list

class Decoder_body(nn.Module):
    def __init__(self):
        super(Decoder_body, self).__init__()
        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0, 0, 0, 0]):
        out_list = []
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out_list.append(out0)

        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(torch.cat((input1[1], out0), dim=1))), inplace=True)
        out_list.append(out1)

        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(torch.cat((input1[2], out1), dim=1))), inplace=True)
        out_list.append(out2)

        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(torch.cat((input1[3], out2), dim=1))), inplace=True)
        out_list.append(out3)
        return out_list
    # def initialize(self):
    #     weight_init(self)

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


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, dilation, padding):
        super(DWConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding, groups=in_channel,
                                dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.DWConv(x)
        out = self.selu(self.bn(x))

        return out


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding, kernels_per_layer):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out

from module.BoundaryCoreConditioner import ConditionExtractor


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        cp_condition_net,
        dim_mults=(1, 2, 4, 8),
        self_condition = True,
        with_time_emb = True,
        residual = False,
    ):
        super(Unet, self).__init__()

        self.condition_extractor = ConditionExtractor(dim, cp_condition_net, dim_mults, False, False, residual)

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

        for i in range(len(in_out_mask)):

            self.body_sup.append(nn.ModuleList([
                # nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                nn.Conv2d(self.side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))

            self.detail_sup.append(nn.ModuleList([
                # nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                nn.Conv2d(self.side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))

            self.side_out_sup.append(nn.ModuleList([
                # nn.Conv2d(dim_in, 128, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(128),
                nn.Conv2d(side_unit_channel, 1, kernel_size=3, stride=1, padding=1)
            ]))

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
                Residual(PreNorm(dim_in, LinearCorssAttention(dim_in, context_in=self.side_unit_channel))),
                # LinearCorssAttention(
                #     dim_out, 4, 32, depth=1, context_dim=dim_out),
                # nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
                # Residual(attn_klass(dim_in, **attn_kwargs)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))


        mid_dim = dims_mask[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearCorssAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # self.cond_mid_block1 = block_klass(512, mid_dim, time_emb_dim = time_dim)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)



        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out_mask), reversed(full_self_attn))):

            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_in*3, dim_in, time_emb_dim = time_dim) if ind < 3 else block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in*2, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearCorssAttention(dim_in, context_in=self.side_unit_channel))),
                # Residual(attn_klass(dim_out, **attn_kwargs)),
                # SpatialTransformer(
                #     dim_out, 4, 32, depth=1, context_dim=dim_out),
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
    def forward(self, input, label_noise_t, time, x_self_cond):

        B,C,H,W, = input.shape
        x = label_noise_t
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(input))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.residual:
            orig_x = input

        x = self.init_conv(x)
        # cond = self.init_conv_cond(input)
        # r = x.clone()

        # label = label_noise_t
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        out_body, out_detail, out_body_detail, cond= self.condition_extractor(input)

        side_for_sup = []
        body_detail_sup = [tensor.clone() for tensor in out_body_detail]
        body_detail_sup = body_detail_sup[::-1]

        body_for_sup = []
        body_sup = [tensor.clone() for tensor in out_body]
        body_sup = body_sup[::-1]

        detail_for_sup = []
        detail_sup = [tensor.clone() for tensor in out_detail]
        detail_sup = detail_sup[::-1]


        for i, ModuleList in enumerate(self.side_out_sup):
            for conv in ModuleList:
                # print(conv)
                side = conv(body_detail_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                side_for_sup.append(side)

        for i, ModuleList in enumerate(self.body_sup):
            for conv in ModuleList:
                # print(conv)
                side = conv(body_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                body_for_sup.append(side)

        for i, ModuleList in enumerate(self.detail_sup):
            for conv in ModuleList:
                # print(conv)
                side = conv(detail_sup[i])
                side = F.interpolate(side, size=[H, W], mode='bilinear')
                detail_for_sup.append(side)

        # label_noise
        out_body_detail_for_label_decoder = [tensor.clone() for tensor in out_body_detail]
        # out_body_detail_for_label_decoder = out_body_detail_for_label_decoder[::-1]

        out_body_for_label_decoder = [tensor.clone() for tensor in out_body]
        # out_body_for_label_decoder = out_body_for_label_decoder[::-1]

        out_detail_for_label_decoder = [tensor.clone() for tensor in out_detail]
        # out_detail_for_label_decoder = out_detail_for_label_decoder[::-1]

        label_noise_side = []
        num = 0
        out_body_detail = out_body_detail[::-1]
        out_body = out_body[::-1]
        out_detail = out_detail[::-1]
        for convnext, convnext2, LinearCorssAttention, downsample in self.downs_label_noise:
            if num == 0:
                if x.shape != out_body_detail[0].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[0], size=(H,W), mode="bilinear")
                x = x + condition

            if num == 1:
                if x.shape != out_body_detail[1].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[1], size=(H,W), mode="bilinear")
                x = x + condition

            if num == 2:
                if x.shape != out_body_detail[2].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[2], size=(H,W), mode="bilinear")

                x = x + torch.cat((condition, condition), dim=1)

            if num == 3:
                if x.shape != out_body_detail[3].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail[3], size=(H,W), mode="bilinear")
                x = x + torch.cat((condition, condition, condition, condition), dim=1)


            x = convnext(x, t)
            label_noise_side.append(x)

            x = convnext2(x, t)
            if num <= 1:
                context = out_body[num]
            if num > 1:
                context = out_detail[num]
            x = LinearCorssAttention(x, context)
            label_noise_side.append(x)
            x = downsample(x)
            num = num + 1

        x = self.mid_block1(x, t)
        cond = self.cond_mid_block1(cond, t)
        if x.shape != cond.shape:
            B, C, H, W, = x.shape
            cond = F.interpolate(cond, size=(H, W), mode="bilinear")
        x = x + cond
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # #invert side out for concat
        # input_side = input_side[::-1]
        # label_noise_side = label_noise_side[::-1]
        num = 0
        for convnext, convnext2, LinearCorssAttention, upsample in self.ups:
            # input_side and label_noise_side CONCAT
            # side_put = torch.cat([input_side.pop(),label_noise_side.pop()],dim=1)
            #
            # condition = out_body_detail_for_label_decoder.pop()
            if num == 0:
                if x.shape != out_body_detail_for_label_decoder[0].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[0], size=(H,W), mode="bilinear")

                condition = torch.cat((condition, condition, condition,
                                           condition), dim=1)
            if num == 1:
                if x.shape != out_body_detail_for_label_decoder[1].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[1], size=(H,W), mode="bilinear")

                condition = torch.cat(
                    (condition, condition), dim=1)
            if num == 2:
                if x.shape != out_body_detail_for_label_decoder[2].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[2], size=(H,W), mode="bilinear")
                condition = condition
            if num == 3:
                if x.shape != out_body_detail_for_label_decoder[3].shape:
                    B, C, H, W, = x.shape
                    condition = F.interpolate(out_body_detail_for_label_decoder[3], size=(H,W), mode="bilinear")
                condition = condition
            # condition = out_body_detail_for_label_decoder.pop() + torch.cat((out_body_for_label_decoder.pop(),out_detail_for_label_decoder.pop()),dim=1)
            # y = label_noise_side.pop()
            # x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = torch.cat((x, label_noise_side.pop()), dim=1)
            x = convnext(x, t)
            x = x + condition
            x = torch.cat((x, label_noise_side.pop()), dim=1)

            x = convnext2(x, t)
            # x = LinearCorssAttention(x)
            if num <= 1:
                x = LinearCorssAttention(x, out_detail_for_label_decoder[num])
            if num > 1:
                x = LinearCorssAttention(x, out_body_for_label_decoder[num])
            x = upsample(x)
            num = num + 1

        if self.residual:
            return self.final_conv(x)
        # x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        # return self.final_conv(x)
        return self.final_conv(x), side_for_sup, body_for_sup, detail_for_sup


if __name__ == '__main__':
    import os


    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        with_time_emb=True,
        residual=False
    ).cuda()

    # save_dict = torch.load('/home/david/model-12.pt')
    # model.load_state_dict(save_dict['model'], strict=True)
    # save_dict = torch.load(
    #     "/home/david/model-12.pt")[
    #     'model']
    # filtered_state_dict = {k: v for k, v in save_dict.items() if 'model' in k}
    # fixed_state_dict = {k.replace('model.', ''): v for k, v in filtered_state_dict.items()}
    # model.load_state_dict(fixed_state_dict, strict=False)

    input_R = torch.randn(1,1,256,256).cuda()
    label_noise_t = torch.randn(1,1,256,256).cuda()
    time = torch.randn(2).cuda()
    X=model(input_R, label_noise_t, time, None)