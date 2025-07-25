import torch
import torch.nn as nn

class Conv(nn.Module):
    # 包含BN和ReLU
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWR(nn.Module):
    def __init__(self, c) -> None:
        super(DWR, self).__init__()

        self.conv_3x3 = Conv(c, c, 3, padding=1)

        self.conv_3x3_d1 = Conv(c, c, 3, padding=1, dilation=1)
        self.conv_3x3_d3 = Conv(c, c, 3, padding=3, dilation=3)
        self.conv_3x3_d5 = Conv(c, c, 3, padding=5, dilation=5)

        self.conv_1x1 = Conv(c * 3, c, 1)

    def forward(self, x):
        x_ = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(x_)
        x2 = self.conv_3x3_d3(x_)
        x3 = self.conv_3x3_d5(x_)

        # 这里确保所有的特征图尺寸匹配
        size = x1.size()[2:]
        x2 = nn.functional.interpolate(x2, size=size, mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(x3, size=size, mode='bilinear', align_corners=False)

        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class DWRSeg_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super(DWRSeg_Conv, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size)

        self.dcnv3 = DWR(out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dcnv3(x)
        x = self.gelu(self.bn(x))
        return x

class Bottleneck_DWRSeg(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super(Bottleneck_DWRSeg, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DWRSeg_Conv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DWRSeg(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super(C2f_DWRSeg, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([Bottleneck_DWRSeg(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))

        # 确保所有特征图尺寸一致
        size = y[0].size()[2:]
        for i in range(1, len(y)):
            y[i] = nn.functional.interpolate(y[i], size=size, mode='bilinear', align_corners=False)

        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))

        # 确保所有特征图尺寸一致
        size = y[0].size()[2:]
        for i in range(1, len(y)):
            y[i] = nn.functional.interpolate(y[i], size=size, mode='bilinear', align_corners=False)

        return self.cv2(torch.cat(y, 1))
