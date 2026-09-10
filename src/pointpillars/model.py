"""Full PointPillars model: PillarFeatureNet -> Scatter -> Backbone -> SSDHead.

Anchors are built lazily on the first forward pass, once the backbone output
resolution is known, and cached thereafter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .anchors import generate_anchors
from .backbone import Backbone
from .head import SSDHead
from .pillars import PillarFeatureNet, Scatter


class PointPillars(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.pc_range = d["pc_range"]
        self.voxel_size = d["voxel_size"]
        self.out_stride = 2
        self.sizes = [tuple(s) for s in m["anchor_sizes"]]
        self.rotations = m["anchor_rotations"]
        self.z_center = m["anchor_z"]
        self.num_classes = m.get("num_classes", 1)
        self.num_anchors = len(self.sizes) * len(self.rotations)

        gx, gy = self.voxel_size
        self.grid_w = int(round((self.pc_range[3] - self.pc_range[0]) / gx))
        self.grid_h = int(round((self.pc_range[4] - self.pc_range[1]) / gy))

        cch = m.get("pillar_channels", 64)
        self.pfn = PillarFeatureNet(9, cch)
        self.scatter = Scatter(self.grid_h, self.grid_w)
        self.backbone = Backbone(in_ch=cch, base=m.get("backbone_base", 64))
        self.head = SSDHead(self.backbone.out_channels, self.num_anchors, self.num_classes)
        self._anchors = None

    def anchors(self, feat_h: int, feat_w: int, device) -> torch.Tensor:
        if self._anchors is None or self._anchors.shape[0] != feat_h * feat_w * self.num_anchors:
            self._anchors = generate_anchors(
                feat_h, feat_w, self.pc_range, self.voxel_size, self.out_stride,
                self.sizes, self.rotations, self.z_center).to(device)
        return self._anchors

    def forward(self, pillars, coords, npoints):
        feats = self.pfn(pillars, npoints)               # (B,P,C)
        canvas = self.scatter(feats, coords, npoints)    # (B,C,H,W)
        x = self.backbone(canvas)                        # (B,C',h,w)
        preds = self.head(x)
        anchors = self.anchors(x.shape[-2], x.shape[-1], x.device)
        return preds, anchors
