"""Detection losses: sigmoid focal (cls) + smooth-L1 (reg) + CE (direction)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss)


def detection_loss(preds, labels, reg_targets, dir_targets, num_classes: int,
                   beta_cls=1.0, beta_reg=2.0, beta_dir=0.2):
    """preds: dict of (B, A, C). labels/targets: (B, A[, 7])."""
    cls = preds["cls"]
    B, A, _ = cls.shape

    pos = labels == 1
    valid = labels >= 0                     # exclude ignore (-1)
    n_pos = pos.sum().clamp(min=1)

    # classification (one-hot over foreground classes; here num_classes=1)
    cls_targets = torch.zeros_like(cls)
    cls_targets[pos] = 1.0
    fl = sigmoid_focal_loss(cls, cls_targets)
    fl = fl * valid.unsqueeze(-1)
    cls_loss = fl.sum() / n_pos

    # regression (smooth L1 on positive anchors)
    if pos.any():
        reg_loss = F.smooth_l1_loss(preds["reg"][pos], reg_targets[pos], reduction="sum") / n_pos
        dir_loss = F.cross_entropy(preds["dir"][pos], dir_targets[pos])
    else:
        reg_loss = preds["reg"].sum() * 0.0
        dir_loss = preds["dir"].sum() * 0.0

    total = beta_cls * cls_loss + beta_reg * reg_loss + beta_dir * dir_loss
    parts = {"cls": float(cls_loss.detach()), "reg": float(reg_loss.detach()),
             "dir": float(dir_loss.detach())}
    return total, parts
