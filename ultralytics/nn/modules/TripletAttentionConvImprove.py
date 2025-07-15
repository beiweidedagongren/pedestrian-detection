import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    # 根据输入的 kernel size 和 dilation 自动计算 padding
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # 标准卷积层定义，包括卷积、批归一化和激活函数
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ZPool(nn.Module):
    # 定义 ZPool 操作，对输入进行最大值池化和均值池化并拼接
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    # 简化的卷积层定义，包括卷积、批归一化和激活函数
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
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
    # 注意力门控模块，包括 ZPool 和卷积操作
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
    # 第二种注意力门控模块，包括 ZPool 和卷积操作
    def __init__(self):
        super(AttentionGate2, self).__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class TripletAttention_Conv(nn.Module):
    # 三重注意力卷积模块，包括多种注意力门控模块和卷积操作
    def __init__(self, c1, c2, k=1, s=1, no_spatial=False):
        super(TripletAttention_Conv, self).__init__()
        self.cw = AttentionGate2()
        self.hc = AttentionGate2()
        self.conv = Conv(c1, c2, k, s)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate(c1, c2, k, s)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.conv(x_out11)

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.conv(x_out21)

        if not self.no_spatial:
            x_hw = self.hw(x)
            x_out = 1 / 3 * (x_hw + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)

        return x_out

class TripletAttention_Conv_w(nn.Module):
    # 使用学习权重的三重注意力卷积模块
    def __init__(self, c1, c2, k=1, s=1):
        super(TripletAttention_Conv_w, self).__init__()
        self.cw = AttentionGate2()
        self.hc = AttentionGate2()
        self.conv = Conv(c1, c2, k, s)
        self.hw = AttentionGate(c1, c2, k, s)
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

        x_hw = self.hw(x)

        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        x_out = w[0] * x_hw + w[1] * x_out11 + w[2] * x_out21

        return x_out

class MultiScaleConv(nn.Module):
    def __init__(self, c1, c2):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(c1, c2, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = x1 + x3 + x5
        x = self.bn(x)
        return self.act(x)

# 改进的TA-Conv模块
class TripletAttention_Conv_w1(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super(TripletAttention_Conv_w1, self).__init__()
        self.cw = AttentionGate2()
        self.hc = AttentionGate2()
        self.conv = Conv(c1, c2, k, s)
        self.hw = AttentionGate(c1, c2, k, s)
        self.multi_scale_conv = MultiScaleConv(c2, c2)  # 新增多尺度特征提取模块
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
        x_out = self.multi_scale_conv(x_out)  # 多尺度特征提取

        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x_out = (weight[0] * x_out + weight[1] * x_out11 + weight[2] * x_out21)

        return x_out
