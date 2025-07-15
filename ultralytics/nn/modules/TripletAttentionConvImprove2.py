import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
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
        self.compress = ZPool()
        self.conv1 = Conv(c1, c2, k, s, p)
        self.conv2 = Conv(2, 1, k, s, p)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv2(x_compress)
        scale = torch.sigmoid(x_out)
        x = self.conv1(x)
        return x * scale

class AttentionGate2(nn.Module):
    def __init__(self):
        super(AttentionGate2, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class PolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x

        spatial_wv = self.sp_wv(x)
        spatial_wq = self.sp_wq(x)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x

        out = spatial_out + channel_out
        return out

class TripletAttention_Conv_w2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, dimension=1):
        super(TripletAttention_Conv_w2, self).__init__()
        self.cw = AttentionGate2()
        self.hc = AttentionGate2()
        self.conv = Conv(c1, c2, k, s, p=None)
        self.hw = AttentionGate(c1, c2, k, s, p=None)
        self.psa = PolarizedSelfAttention(channel=c2)
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.conv(x_out11)

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.conv(x_out21)

        x_out = self.hw(x)

        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x_out = (weight[0] * x_out + weight[1] * x_out11 + weight[2] * x_out21)
        x_out = self.psa(x_out)
        return x_out
