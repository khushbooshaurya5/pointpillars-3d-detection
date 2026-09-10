"""Anchor generation, target assignment, and box encode/decode.

Boxes are 7-DoF: ``[x, y, z, w, l, h, theta]`` in the LiDAR frame. Target
assignment uses a **BEV IoU** computed on axis-aligned footprints (a common,
readable simplification of rotated IoU that keeps the repo dependency-free);
swap in rotated IoU for best accuracy. Encoding follows the SECOND residual
codec used by PointPillars.
"""

from __future__ import annotations

import numpy as np
import torch


def generate_anchors(feat_h: int, feat_w: int, pc_range, voxel_size, out_stride: int,
                     sizes, rotations, z_center: float) -> torch.Tensor:
    x_min, y_min = pc_range[0], pc_range[1]
    vx, vy = voxel_size
    step_x = vx * out_stride
    step_y = vy * out_stride
    xs = x_min + (np.arange(feat_w) + 0.5) * step_x
    ys = y_min + (np.arange(feat_h) + 0.5) * step_y
    xx, yy = np.meshgrid(xs, ys)  # (H, W)

    anchors = []
    for (w, l, h) in sizes:
        for rot in rotations:
            a = np.zeros((feat_h, feat_w, 7), np.float32)
            a[..., 0] = xx
            a[..., 1] = yy
            a[..., 2] = z_center
            a[..., 3] = w
            a[..., 4] = l
            a[..., 5] = h
            a[..., 6] = rot
            anchors.append(a.reshape(-1, 7))
    # interleave per-location so reshape back to (H,W,n_anchor,7) is consistent
    anchors = np.stack(anchors, axis=1).reshape(-1, 7)  # (H*W*n_anchor, 7)
    return torch.from_numpy(anchors)


def _bev_iou_axis_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of axis-aligned footprints. a:(Na,7) b:(Nb,7) -> (Na,Nb)."""
    def to_box(x):
        cx, cy, w, l = x[:, 0], x[:, 1], x[:, 3], x[:, 4]
        return torch.stack([cx - w / 2, cy - l / 2, cx + w / 2, cy + l / 2], 1)
    A, B = to_box(a), to_box(b)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    lt = torch.max(A[:, None, :2], B[None, :, :2])
    rb = torch.min(A[:, None, 2:], B[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def encode(gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    dx = (gt[:, 0] - anchors[:, 0]) / diag
    dy = (gt[:, 1] - anchors[:, 1]) / diag
    dz = (gt[:, 2] - anchors[:, 2]) / anchors[:, 5]
    dw = torch.log(gt[:, 3] / anchors[:, 3])
    dl = torch.log(gt[:, 4] / anchors[:, 4])
    dh = torch.log(gt[:, 5] / anchors[:, 5])
    dtheta = gt[:, 6] - anchors[:, 6]
    return torch.stack([dx, dy, dz, dw, dl, dh, dtheta], 1)


def decode(reg: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = reg[:, 0] * diag + anchors[:, 0]
    y = reg[:, 1] * diag + anchors[:, 1]
    z = reg[:, 2] * anchors[:, 5] + anchors[:, 2]
    w = torch.exp(reg[:, 3]) * anchors[:, 3]
    l = torch.exp(reg[:, 4]) * anchors[:, 4]
    h = torch.exp(reg[:, 5]) * anchors[:, 5]
    theta = reg[:, 6] + anchors[:, 6]
    return torch.stack([x, y, z, w, l, h, theta], 1)


def assign_targets(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                   pos_thr: float = 0.6, neg_thr: float = 0.45):
    """Return labels (A,), reg_targets (A,7), dir_targets (A,)."""
    A = anchors.shape[0]
    labels = torch.zeros(A, dtype=torch.long)      # 0 = negative
    reg_targets = torch.zeros(A, 7)
    dir_targets = torch.zeros(A, dtype=torch.long)
    if gt_boxes.numel() == 0:
        return labels, reg_targets, dir_targets

    iou = _bev_iou_axis_aligned(anchors, gt_boxes)  # (A, G)
    max_iou, argmax = iou.max(dim=1)
    labels[max_iou < neg_thr] = 0
    labels[(max_iou >= neg_thr) & (max_iou < pos_thr)] = -1   # ignore
    pos = max_iou >= pos_thr
    # ensure each gt has at least one positive anchor
    gt_best = iou.argmax(dim=0)
    pos[gt_best] = True

    labels[pos] = 1
    reg_targets[pos] = encode(gt_boxes[argmax[pos]], anchors[pos])
    dir_targets[pos] = (gt_boxes[argmax[pos], 6] > 0).long()
    return labels, reg_targets, dir_targets
