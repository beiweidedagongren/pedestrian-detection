import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ModifiedSPPF(nn.Module):
    """Modified Spatial Pyramid Pooling - Fast (SPPF) layer with average pooling and direct input concatenation."""

    def __init__(self, c1, c2, k1=5, k2=9):
        """
        Initializes the Modified SPPF layer with given input/output channels and two kernel sizes for average pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 3 + c1, c2, 1, 1)  # Adjusted for concatenation with original input
        self.ap1 = nn.AvgPool2d(kernel_size=k1, stride=1, padding=k1 // 2)
        self.ap2 = nn.AvgPool2d(kernel_size=k2, stride=1, padding=k2 // 2)

    def forward(self, x):
        """Forward pass through the Modified SPPF layer."""
        original = x  # Save original input for later concatenation
        x = self.cv1(x)
        y1 = self.ap1(x)
        y2 = self.ap2(x)

        # Concatenate along the channel dimension
        concat_out = torch.cat((x, y1, y2, original), 1)
        return self.cv2(concat_out)
