import torch
from torch import nn
from torch.nn import init

class EMCA(nn.Module):
    def __init__(self, groups, kernel_size=3):
        super().__init__()
        self.groups = groups
        self.kernel_size = kernel_size
        self.init_weights()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_max = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        if c % self.groups != 0:
            self.groups = min(c, self.groups)
            if c % self.groups != 0:
                raise ValueError(f'Number of channels ({c}) must be divisible by the number of groups ({self.groups}) after adjustment')

        x = x.view(b * self.groups, -1, h, w)
        y1 = self.gap(x)
        y2 = self.gap_max(x)
        y = y1 + y2
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        out = x * y.expand_as(x)
        out = out.view(b, c, h, w)
        return out
