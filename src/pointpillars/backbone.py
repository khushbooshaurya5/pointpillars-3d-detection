"""2D backbone + multi-scale upsampling neck (SECOND/PointPillars style)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch, out_ch, num_convs, stride):
    layers = [nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    for _ in range(num_convs):
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Backbone(nn.Module):
    def __init__(self, in_ch: int = 64, base: int = 64) -> None:
        super().__init__()
        self.b1 = _block(in_ch, base, 3, stride=2)          # /2
        self.b2 = _block(base, base * 2, 5, stride=2)       # /4
        self.b3 = _block(base * 2, base * 4, 5, stride=2)   # /8
        # upsample each stage to the /2 resolution and concat
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base, base * 2, 1, stride=1, bias=False),
                                 nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2, bias=False),
                                 nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 4, stride=4, bias=False),
                                 nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True))
        self.out_channels = base * 6

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        u1 = self.up1(x1)
        u2 = self.up2(x2)
        u3 = self.up3(x3)
        # align spatial dims to u1 (defensive against odd sizes)
        h, w = u1.shape[-2:]
        u2 = u2[..., :h, :w]
        u3 = u3[..., :h, :w]
        return torch.cat([u1, u2, u3], dim=1)
