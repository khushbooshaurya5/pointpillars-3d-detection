"""Draw a BEV image with predicted (green) and GT (red) box footprints."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .dataset import SyntheticKITTI3D, KITTIObject, collate
from .inference import postprocess
from .model import PointPillars
from .train import load_config


def _draw_box(img, box, pc_range, res, color):
    x0, y0 = pc_range[0], pc_range[1]
    cx, cy, w, l, theta = box[0], box[1], box[3], box[4], box[6]
    corners = np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = corners @ R.T + np.array([cx, cy])
    px = ((corners[:, 0] - x0) / res).astype(int)
    py = ((corners[:, 1] - y0) / res).astype(int)
    H, W = img.shape[:2]
    for i in range(4):
        _line(img, px[i], py[i], px[(i + 1) % 4], py[(i + 1) % 4], color, H, W)


def _line(img, x0, y0, x1, y1, color, H, W):
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0, 1, n * 2):
        x = int(x0 + t * (x1 - x0)); y = int(y0 + t * (y1 - y0))
        if 0 <= y < H and 0 <= x < W:
            img[y, x] = color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="bev.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    ds = SyntheticKITTI3D(d, length=4) if args.synthetic else KITTIObject(d["root"], d, "val")
    batch = collate([ds[args.index]])

    model = PointPillars(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    with torch.no_grad():
        preds, anchors = model(batch["pillars"].float(), batch["coords"], batch["npoints"])
        boxes, scores = postprocess(preds, anchors, cfg["train"].get("score_thr", 0.3))[0]

    res = 0.1
    x0, y0, _, x1, y1, _ = d["pc_range"]
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    img = np.zeros((H, W, 3), np.uint8)
    for gt in batch["gt"][0].numpy():
        _draw_box(img, gt, d["pc_range"], res, (255, 60, 60))
    for b in boxes.numpy():
        _draw_box(img, b, d["pc_range"], res, (60, 255, 60))

    try:
        from PIL import Image
        Image.fromarray(img).save(args.out)
        print(f"saved {args.out}  ({len(boxes)} predictions, {len(batch['gt'][0])} gt)")
    except ImportError:
        np.save(args.out + ".npy", img)


if __name__ == "__main__":
    main()
