"""Train PointPillars.

Smoke test (CPU, synthetic)::

    python -m pointpillars.train --config configs/smoke.yaml --synthetic

Full KITTI training (GPU)::

    python -m pointpillars.train --config configs/kitti.yaml
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from .anchors import assign_targets
from .dataset import KITTIObject, SyntheticKITTI3D, collate
from .losses import detection_loss
from .model import PointPillars


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loader(cfg, synthetic):
    d = cfg["data"]
    if synthetic:
        ds = SyntheticKITTI3D(d, length=d.get("synth_len", 8))
    else:
        ds = KITTIObject(d["root"], d, "train")
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                      collate_fn=collate, num_workers=cfg["train"].get("num_workers", 0),
                      drop_last=True)


def build_targets(anchors, gts, cfg):
    labels, regs, dirs = [], [], []
    for gt in gts:
        lb, rg, dr = assign_targets(anchors.cpu(), gt,
                                    cfg["train"]["pos_iou"], cfg["train"]["neg_iou"])
        labels.append(lb); regs.append(rg); dirs.append(dr)
    return (torch.stack(labels).to(anchors.device),
            torch.stack(regs).to(anchors.device),
            torch.stack(dirs).to(anchors.device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    loader = make_loader(cfg, args.synthetic)
    model = PointPillars(cfg).to(device)
    print(f"[pointpillars] device={device} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M batches={len(loader)}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 0.01))

    epochs = args.epochs or cfg["train"]["epochs"]
    for epoch in range(epochs):
        model.train(); running = 0.0
        for batch in loader:
            pillars = batch["pillars"].to(device).float()
            coords = batch["coords"].to(device)
            npoints = batch["npoints"].to(device)
            preds, anchors = model(pillars, coords, npoints)
            labels, regs, dirs = build_targets(anchors, batch["gt"], cfg)
            opt.zero_grad()
            loss, parts = detection_loss(preds, labels, regs, dirs, model.num_classes)
            loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {epoch+1}/{epochs}  loss={running/max(1,len(loader)):.4f}  "
              f"cls={parts['cls']:.3f} reg={parts['reg']:.3f} dir={parts['dir']:.3f}")
        torch.save({"model": model.state_dict(), "config": cfg},
                   os.path.join(args.out, "last.pth"))
    print("[pointpillars] done.")


if __name__ == "__main__":
    main()
