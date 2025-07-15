import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AdaptiveChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AdaptiveChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        max_out = self.global_max_pool(x)
        avg_out = self.conv2(F.relu(self.conv1(avg_out)))
        max_out = self.conv2(F.relu(self.conv1(max_out)))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class DilatedConvModule(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        return self.conv(x)


class AdaptiveWeighting(nn.Module):
    def forward(self, weights, feature_maps, epsilon=0.0001):
        normalized_weights = weights / (torch.sum(weights) + epsilon)
        weighted_feature_maps = [normalized_weights[i] * feature_maps[i] for i in range(len(feature_maps))]
        return torch.stack(weighted_feature_maps, dim=0)


class AFPBiFPN(nn.Module):
    def __init__(self, in_channels, length):
        super(AFPBiFPN, self).__init__()
        self.attention_modules = nn.ModuleList([AdaptiveChannelAttention(in_channels) for _ in range(length)])
        self.weighting = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        attention_maps = [self.attention_modules[i](x[i]) for i in range(len(x))]
        weighted_maps = [self.weighting[i] * attention_maps[i] for i in range(len(attention_maps))]
        return torch.sum(torch.stack(weighted_maps, dim=0), dim=0)



