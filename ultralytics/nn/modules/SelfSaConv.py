import torch
import torch.nn as nn
import torch.nn.functional as F

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
class SelfAttention(nn.Module):
    def __init__(self, in_channels, k=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.k = k
        self.query_conv = nn.Conv2d(in_channels, in_channels // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)

        attention = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SelfAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super(SelfAttentionConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.self_attention = SelfAttention(out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.self_attention(x)
        sa = self.spatial_attention(x)
        return x * sa

# Example of using the SelfAttentionConv module
class CombinedAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super(CombinedAttentionConv, self).__init__()
        self.conv1 = SelfAttentionConv(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = Conv(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
