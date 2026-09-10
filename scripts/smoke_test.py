"""End-to-end smoke test: voxelize -> full model -> loss -> backward (CPU)."""

from __future__ import annotations

import sys

import torch

from pointpillars.anchors import assign_targets
from pointpillars.dataset import SyntheticKITTI3D, collate
from pointpillars.inference import postprocess
from pointpillars.losses import detection_loss
from pointpillars.model import PointPillars

CFG = {
    "data": {"pc_range": [0.0, -20.0, -3.0, 40.0, 20.0, 1.0], "voxel_size": [0.32, 0.32],
             "max_points": 16, "max_pillars": 3000, "synth_len": 6},
    "model": {"num_classes": 1, "pillar_channels": 32, "backbone_base": 16,
              "anchor_sizes": [[1.6, 3.9, 1.5]], "anchor_rotations": [0.0, 1.57], "anchor_z": -1.0},
    "train": {"pos_iou": 0.4, "neg_iou": 0.3, "score_thr": 0.2},
}


def main() -> None:
    device = torch.device("cpu")
    ds = SyntheticKITTI3D(CFG["data"], length=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate)
    model = PointPillars(CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"[smoke] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    first, last = None, None
    for _ in range(2):
        for batch in loader:
            preds, anchors = model(batch["pillars"].float(), batch["coords"], batch["npoints"])
            labels, regs, dirs = [], [], []
            for gt in batch["gt"]:
                lb, rg, dr = assign_targets(anchors, gt, 0.4, 0.3)
                labels.append(lb); regs.append(rg); dirs.append(dr)
            labels, regs, dirs = torch.stack(labels), torch.stack(regs), torch.stack(dirs)
            opt.zero_grad()
            loss, parts = detection_loss(preds, labels, regs, dirs, 1)
            loss.backward(); opt.step()
            last = loss.item()
            if first is None:
                first = last
    print(f"[smoke] anchors={anchors.shape[0]}  loss {first:.4f} -> {last:.4f}  {parts}")

    # inference path
    model.eval()
    with torch.no_grad():
        preds, anchors = model(batch["pillars"].float(), batch["coords"], batch["npoints"])
        res = postprocess(preds, anchors, score_thr=0.0)
    print(f"[smoke] decoded {res[0][0].shape[0]} boxes for sample 0")
    assert torch.isfinite(torch.tensor(last)), "loss diverged"
    print("[smoke] PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
