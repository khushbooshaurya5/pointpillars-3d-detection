"""Simple BEV detection metrics: precision / recall / AP at an IoU threshold.

This is a lightweight, dependency-free scorer for sanity-checking training. For
official KITTI AP (11/40-point, per-difficulty), export predictions in the
KITTI format and use the official devkit.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .anchors import _bev_iou_axis_aligned
from .dataset import KITTIObject, SyntheticKITTI3D, collate
from .inference import postprocess
from .model import PointPillars
from .train import load_config


def evaluate_ap(all_preds, all_gts, iou_thr: float = 0.5):
    """all_preds: list of (boxes, scores); all_gts: list of boxes."""
    scored = []
    n_gt = 0
    for (boxes, scores), gts in zip(all_preds, all_gts):
        n_gt += gts.shape[0]
        matched = torch.zeros(gts.shape[0], dtype=torch.bool)
        order = scores.argsort(descending=True)
        for idx in order:
            if gts.shape[0] == 0:
                scored.append((scores[idx].item(), 0)); continue
            iou = _bev_iou_axis_aligned(boxes[idx:idx + 1], gts)[0]
            best = iou.argmax()
            if iou[best] >= iou_thr and not matched[best]:
                matched[best] = True
                scored.append((scores[idx].item(), 1))
            else:
                scored.append((scores[idx].item(), 0))
    if not scored or n_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": n_gt}
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([1 - s[1] for s in scored])
    recall = tp / max(n_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    # 11-point interpolated AP
    ap = np.mean([precision[recall >= t].max() if np.any(recall >= t) else 0.0
                  for t in np.linspace(0, 1, 11)])
    return {"AP": float(ap), "precision": float(precision[-1]),
            "recall": float(recall[-1]), "n_gt": n_gt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]
    ds = SyntheticKITTI3D(d, length=4) if args.synthetic else KITTIObject(d["root"], d, "val")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate)

    model = PointPillars(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch in loader:
            preds, anchors = model(batch["pillars"].to(device).float(),
                                   batch["coords"].to(device), batch["npoints"].to(device))
            for (b, s) in postprocess(preds, anchors, cfg["train"].get("score_thr", 0.3)):
                all_preds.append((b.cpu(), s.cpu()))
            all_gts.extend(batch["gt"])

    print(evaluate_ap(all_preds, all_gts, args.iou))


if __name__ == "__main__":
    main()
