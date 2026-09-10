"""Pillar encoding: voxelize a point cloud into pillars, learn per-pillar
features with a mini-PointNet, and scatter them back to a 2D pseudo-image.

This is the front-end of PointPillars (Lang et al., CVPR 2019). Voxelization is
done in numpy (in the dataset) so batching is trivial; the learnable
``PillarFeatureNet`` and ``Scatter`` run on the GPU.

Per-point feature (D = 9): ``[x, y, z, r, xc, yc, zc, xp, yp]`` where
``(xc,yc,zc)`` is the offset to the pillar's point-mean and ``(xp,yp)`` the
offset to the pillar's geometric center.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def voxelize(points: np.ndarray, pc_range, voxel_size, max_points: int,
             max_pillars: int):
    """Return (pillars, coords, npoints) fixed-size arrays.

    pillars: (max_pillars, max_points, 9) float32
    coords:  (max_pillars, 2) int64  -> (x_idx, y_idx) in the BEV grid
    npoints: (max_pillars,)   int64  -> valid points per pillar
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    vx, vy = voxel_size
    grid_w = int(round((x_max - x_min) / vx))
    grid_h = int(round((y_max - y_min) / vy))

    p = points
    keep = ((p[:, 0] >= x_min) & (p[:, 0] < x_max) &
            (p[:, 1] >= y_min) & (p[:, 1] < y_max) &
            (p[:, 2] >= z_min) & (p[:, 2] < z_max))
    p = p[keep]
    if p.shape[0] == 0:
        return (np.zeros((max_pillars, max_points, 9), np.float32),
                np.zeros((max_pillars, 2), np.int64),
                np.zeros((max_pillars,), np.int64))

    xi = ((p[:, 0] - x_min) / vx).astype(np.int64)
    yi = ((p[:, 1] - y_min) / vy).astype(np.int64)
    xi = np.clip(xi, 0, grid_w - 1)
    yi = np.clip(yi, 0, grid_h - 1)
    keys = yi * grid_w + xi

    order = np.argsort(keys)
    p, keys, xi, yi = p[order], keys[order], xi[order], yi[order]
    uniq, start = np.unique(keys, return_index=True)

    pillars = np.zeros((max_pillars, max_points, 9), np.float32)
    coords = np.zeros((max_pillars, 2), np.int64)
    npoints = np.zeros((max_pillars,), np.int64)

    n_pillars = min(len(uniq), max_pillars)
    for i in range(n_pillars):
        s = start[i]
        e = start[i + 1] if i + 1 < len(start) else len(keys)
        pts = p[s:e][:max_points]
        m = pts.shape[0]
        center = pts[:, :3].mean(0)
        px = x_min + (xi[s] + 0.5) * vx
        py = y_min + (yi[s] + 0.5) * vy
        feat = np.zeros((m, 9), np.float32)
        feat[:, :4] = pts[:, :4]
        feat[:, 4:7] = pts[:, :3] - center
        feat[:, 7] = pts[:, 0] - px
        feat[:, 8] = pts[:, 1] - py
        pillars[i, :m] = feat
        coords[i] = [xi[s], yi[s]]
        npoints[i] = m
    return pillars, coords, npoints


class PillarFeatureNet(nn.Module):
    def __init__(self, in_ch: int = 9, out_ch: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, pillars: torch.Tensor, npoints: torch.Tensor) -> torch.Tensor:
        # pillars: (B, P, N, 9) -> pillar features (B, P, C)
        B, P, N, D = pillars.shape
        x = self.linear(pillars)                          # (B,P,N,C)
        C = x.shape[-1]
        x = self.bn(x.view(-1, C)).view(B, P, N, C)
        x = torch.relu(x)
        # mask padded points
        mask = torch.arange(N, device=pillars.device)[None, None, :] < npoints[..., None]
        x = x * mask.unsqueeze(-1)
        return x.max(dim=2)[0]                             # (B,P,C)


class Scatter(nn.Module):
    def __init__(self, grid_h: int, grid_w: int) -> None:
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w

    def forward(self, pillar_feats: torch.Tensor, coords: torch.Tensor,
                npoints: torch.Tensor) -> torch.Tensor:
        # pillar_feats: (B,P,C)  coords: (B,P,2) [x_idx,y_idx]
        B, P, C = pillar_feats.shape
        canvas = pillar_feats.new_zeros(B, C, self.grid_h, self.grid_w)
        valid = npoints > 0
        for b in range(B):
            v = valid[b]
            xs = coords[b, v, 0].long().clamp(0, self.grid_w - 1)
            ys = coords[b, v, 1].long().clamp(0, self.grid_h - 1)
            canvas[b, :, ys, xs] = pillar_feats[b, v].t()
        return canvas
