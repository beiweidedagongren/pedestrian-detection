import torch
import torch.nn as nn
import torch.nn.functional as F

class EFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EFPN, self).__init__()
        # 通道整合
        self.reduce_conv1 = nn.Conv2d(in_channels[0], out_channels, 1, 1)
        self.reduce_conv2 = nn.Conv2d(in_channels[1], out_channels, 1, 1)
        self.reduce_conv3 = nn.Conv2d(in_channels[2], out_channels, 1, 1)

        # 融合特征
        self.fuse_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fuse_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fuse_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        p3, p4, p5 = x

        # 通道缩减
        p3_reduced = self.reduce_conv1(p3)
        p4_reduced = self.reduce_conv2(p4)
        p5_reduced = self.reduce_conv3(p5)

        # 融合 P5 和 P4
        p4_upsampled = self.upsample(p5_reduced)
        p4_fused = self.fuse_conv1(p4_reduced + p4_upsampled)

        # 融合 P4 和 P3
        p3_upsampled = self.upsample(p4_fused)
        p3_fused = self.fuse_conv2(p3_reduced + p3_upsampled)

        # 融合 P3 到 P4
        p3_downsampled = self.downsample(p3_fused)
        p4_fused_2 = self.fuse_conv3(p4_fused + p3_downsampled)

        # 返回增强特征金字塔
        return p3_fused, p4_fused_2, p5_reduced


