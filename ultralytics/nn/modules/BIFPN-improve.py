import torch.nn as nn
import torch
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.max(x, dim=(2, 3), keepdim=True)[0]
        scale = self.sigmoid(self.conv2(self.conv1(avg_out + max_out)))
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


class EnhancedBiFPN(nn.Module):
    def __init__(self, in_channels, length):
        super(EnhancedBiFPN, self).__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = Swish()
        self.epsilon = 0.0001
        self.attention_modules = nn.ModuleList([AttentionModule(in_channels) for _ in range(length)])
        self.dilated_conv_modules = nn.ModuleList([DilatedConvModule(in_channels) for _ in range(length)])
        self.adaptive_weighting = AdaptiveWeighting()

    def forward(self, x):
        attention_maps = [self.attention_modules[i](x[i]) for i in range(len(x))]
        dilated_feature_maps = [self.dilated_conv_modules[i](attention_maps[i]) for i in range(len(attention_maps))]
        result = self.adaptive_weighting(self.weight, dilated_feature_maps, self.epsilon)
        result = torch.sum(result, dim=0)
        return result



