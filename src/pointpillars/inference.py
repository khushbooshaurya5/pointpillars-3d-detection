"""Decode network outputs to boxes + score-threshold + BEV NMS."""

from __future__ import annotations

import torch

from .anchors import _bev_iou_axis_aligned, decode


def bev_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.2,
            max_keep: int = 100):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < max_keep:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = _bev_iou_axis_aligned(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def postprocess(preds, anchors, score_thr: float = 0.3, iou_thr: float = 0.2):
    """Return list over batch of (boxes (K,7), scores (K,))."""
    cls = preds["cls"].sigmoid()            # (B, A, num_classes)
    reg = preds["reg"]
    B = cls.shape[0]
    results = []
    for b in range(B):
        scores, _ = cls[b].max(dim=1)
        mask = scores > score_thr
        if mask.sum() == 0:
            results.append((torch.zeros(0, 7), torch.zeros(0)))
            continue
        boxes = decode(reg[b][mask], anchors[mask])
        sc = scores[mask]
        keep = bev_nms(boxes, sc, iou_thr)
        results.append((boxes[keep], sc[keep]))
    return results
