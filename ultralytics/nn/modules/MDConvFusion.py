import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MDConvFusion']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class DilatedConv(nn.Module):
    def __init__(self, c1, c2, dilation):
        super().__init__()
        self.conv = Conv(c1, c2, k=3, d=dilation, act=Conv.default_act)

    def forward(self, x):
        return self.conv(x)

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MDConvFusion(nn.Module):
    def __init__(self, channels, dilations=[1, 2, 3]):
        super().__init__()
        self.gscs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])
        self.dilated_convs = nn.ModuleList([
            DilatedConv(channels[0], channels[0], dilation) for dilation in dilations
        ])
        self.swish = swish()
        self.weight = nn.Parameter(torch.ones(len(dilations), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, xs):
        target_size = xs[0].shape[2:]
        processed_features = []

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            processed_features.append(self.gscs[i](x))

        combined_features = torch.stack(processed_features, dim=0)
        combined_features = torch.sum(combined_features, dim=0) / len(processed_features)

        dilated_features = [conv(combined_features) for conv in self.dilated_convs]
        dilated_features = torch.stack(dilated_features, dim=0)

        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)  # 权重归一化处理
        weighted_features = [weights[i] * dilated_features[i] for i in range(len(dilated_features))]
        stacked_features = torch.stack(weighted_features, dim=0)
        result = torch.sum(stacked_features, dim=0)

        return result

# Example of integrating MDConvFusion into a network
class ExampleNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExampleNet, self).__init__()
        self.mdconvfusion = MDConvFusion(in_channels)

    def forward(self, x):
        return self.mdconvfusion(x)
