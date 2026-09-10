"""SSD-style detection head: classification, box regression, direction."""

from __future__ import annotations

import torch
import torch.nn as nn


class SSDHead(nn.Module):
    def __init__(self, in_ch: int, num_anchors: int, num_classes: int = 1) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.cls = nn.Conv2d(in_ch, num_anchors * num_classes, 1)
        self.reg = nn.Conv2d(in_ch, num_anchors * 7, 1)
        self.dir = nn.Conv2d(in_ch, num_anchors * 2, 1)

    def forward(self, x):
        B = x.shape[0]

        def flat(t, c):
            # (B, A*c, H, W) -> (B, H*W*A, c) matching anchor ordering
            t = t.permute(0, 2, 3, 1).contiguous()
            return t.view(B, -1, c)

        return {
            "cls": flat(self.cls(x), self.num_classes),
            "reg": flat(self.reg(x), 7),
            "dir": flat(self.dir(x), 2),
        }
