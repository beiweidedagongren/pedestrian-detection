import math

import torch
import torch.nn as nn


import torch
from torch import nn
from torch.nn import functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ZPool(nn.Module):#(k,3,h,w)-->(k,2,h,w)
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()#(k,3,h,w)-->(k,2,h,w)
        # kernel_size = 3
        # self.conv_for_pool = BasicConv(2, 1, kernel_size, stride=2, padding=(kernel_size - 1) // 2, relu=False)
        # self.conv_for_x = Conv(c1, c2, k, s, p, act=True)
        self.conv1 = Conv(c1, c2, k, s, p)
        self.conv2 = Conv(2, 1, k, s, p)
    def forward(self, x):
        x_compress = self.compress(x)#(k,3,h,w)-->(k,2,h,w)
        x_out = self.conv2(x_compress)#(k,2,h,w)-->(k,1,h,w)
        scale = torch.sigmoid(x_out)#(k,2,h,w)-->(k,1,h,w)
        x = self.conv1(x)
        return x * scale
class AttentionGate2(nn.Module):
    def __init__(self):
        super(AttentionGate2, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )#stride = 1 This is the official calculation for Japanese padding.

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)#sigmoid Proportionally increased use of force after use
        return x * scale  # Special expedition after the final change
class TripletAttention_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, no_spatial=False):
        super(TripletAttention_Conv, self).__init__()
        self.cw = AttentionGate2()#（6，64.320，320）
        self.hc = AttentionGate2()#（6，64.320，320）
        self.conv = Conv(c1, c2, k, s, p=None)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate(c1, c2, k, s, p=None)#（6，64.320，320）

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()#（6，64.320，320）-->（64，6.320，320）
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.conv(x_out11)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.conv(x_out21)
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
#"TripletAttention_Conv_w" is the Ta-conv module that uses learned weights.
class TripletAttention_Conv_w(nn.Module):
    def __init__(self, c1, c2, k=1, s=1,dimension=1):
        super(TripletAttention_Conv_w, self).__init__()
        self.cw = AttentionGate2()#（6，64.320，320）
        self.hc = AttentionGate2()#（6，64.320，320）
        self.conv = Conv(c1, c2, k, s, p=None)
        self.hw = AttentionGate(c1, c2, k, s, p=None)#（6，64.320，320）
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()#（6，64.320，320）-->（64，6.320，320）
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.conv(x_out11)

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.conv(x_out21)

        x_out = self.hw(x)

        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # Unify the progress of the general manager
        x_out = (weight[0] * x_out + weight[1] * x_out11 + weight[2] * x_out21)
        return x_out
#endregion
