# PointPillars — 3D Object Detection on KITTI

### ▶ Live demo: **https://khushbooshaurya5.github.io/khushboo-portfolio-projects/demos/pointpillars/**

A compact, **from-scratch** implementation of PointPillars (Lang et al., CVPR
2019): the fast, industry-standard LiDAR 3D detector that turns a point cloud
into a BEV pseudo-image and runs a 2D SSD detector on it. Predicts 7-DoF
oriented 3D boxes `[x, y, z, w, l, h, θ]` for the Car class.

## Architecture

```
points ─▶ voxelize into pillars ─▶ PillarFeatureNet (mini-PointNet) ─▶ scatter to BEV pseudo-image
       ─▶ 2D backbone + multi-scale upsample neck ─▶ SSD head {cls, box, dir} ─▶ decode + BEV-NMS ─▶ 3D boxes
```

| Component | File |
|-----------|------|
| Pillar voxelization + `PillarFeatureNet` + `Scatter` | `pillars.py` |
| 2D backbone + upsampling neck | `backbone.py` |
| Anchors, BEV-IoU assignment, box encode/decode | `anchors.py` |
| SSD head (cls / reg / dir) | `head.py` |
| Focal + smooth-L1 + direction loss | `losses.py` |
| Decode + BEV NMS | `inference.py` |

## Honest scope notes
- **BEV IoU is axis-aligned** for anchor matching and NMS (a readable
  simplification of rotated IoU). Swap in rotated IoU (e.g. a CUDA op) for the
  last few AP points.
- Single-class (Car) by default; multi-class is a config + `CLASS_MAP` change.
- The built-in evaluator reports a lightweight BEV AP/precision/recall for
  sanity-checking; use the official KITTI devkit for benchmark AP.

These are deliberate, documented trade-offs — the point of the repo is a clear,
correct, end-to-end pipeline you can read in an afternoon.

## Quickstart

```bash
pip install -e .

# End-to-end smoke test (CPU, synthetic point clouds with box clusters)
python scripts/smoke_test.py

# Synthetic training + eval + BEV visualization
python -m pointpillars.train --config configs/smoke.yaml --synthetic
python -m pointpillars.evaluate --config configs/smoke.yaml --synthetic --checkpoint checkpoints/last.pth --iou 0.25
python -m pointpillars.visualize --config configs/smoke.yaml --synthetic --checkpoint checkpoints/last.pth --out bev.png

# Full KITTI training (GPU) — see scripts/DATASET.md
python -m pointpillars.train --config configs/kitti.yaml
```

## Results (KITTI val, Car — fill after training)

| Model | BEV AP@0.5 | 3D AP@0.5 | Notes |
|-------|-----------|-----------|-------|
| this repo | _TBD_ | _TBD_ | axis-aligned IoU assignment |

> The original PointPillars reaches ~89 BEV / ~79 3D AP (moderate, Car). This
> compact re-implementation targets clarity over SOTA; rotated IoU + longer
> schedule close most of the gap.

## Roadmap
- [ ] Rotated BEV IoU + rotated NMS
- [ ] Multi-class (Pedestrian, Cyclist) anchors
- [ ] KITTI-format export for the official devkit

## References
- Lang et al., *PointPillars* (CVPR 2019)
- Yan et al., *SECOND* (Sensors 2018) — the box codec and sparse-conv lineage

## License
MIT © Khushboo Kumari
