"""KITTI 3D object-detection dataset and a synthetic drop-in.

KITTI layout::

    <root>/velodyne/000000.bin     # float32 [x,y,z,r]
    <root>/label_2/000000.txt      # KITTI object labels (camera frame)
    <root>/calib/000000.txt        # P2, R0_rect, Tr_velo_to_cam

Labels are converted from the camera frame to the LiDAR frame and expressed as
7-DoF boxes ``[x, y, z, w, l, h, theta]`` (z at the box center).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .pillars import voxelize

CLASS_MAP = {"Car": 0, "Van": 0}  # single-class (car) detector by default


def _read_calib(path: str):
    d = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    R0 = d["R0_rect"].reshape(3, 3)
    Tr = d["Tr_velo_to_cam"].reshape(3, 4)
    Tr4 = np.eye(4); Tr4[:3, :] = Tr
    R04 = np.eye(4); R04[:3, :3] = R0
    return R04, Tr4


def _cam_to_velo(xyz_cam, R04, Tr4):
    inv = np.linalg.inv(R04 @ Tr4)
    hom = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1))], 1)
    return (inv @ hom.T).T[:, :3]


def _read_labels(path: str, R04, Tr4) -> np.ndarray:
    boxes = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0] not in CLASS_MAP:
                continue
            h, w, l = float(p[8]), float(p[9]), float(p[10])
            xc, yc, zc = float(p[11]), float(p[12]), float(p[13])
            ry = float(p[14])
            center_velo = _cam_to_velo(np.array([[xc, yc, zc]]), R04, Tr4)[0]
            center_velo[2] += h / 2  # cam gives bottom center -> box center
            theta = -ry - np.pi / 2
            boxes.append([center_velo[0], center_velo[1], center_velo[2], w, l, h, theta])
    return np.array(boxes, np.float32).reshape(-1, 7)


class KITTIObject(Dataset):
    def __init__(self, root: str, cfg_data: dict, split: str = "train") -> None:
        self.root = root
        self.pc_range = cfg_data["pc_range"]
        self.voxel_size = cfg_data["voxel_size"]
        self.max_points = cfg_data["max_points"]
        self.max_pillars = cfg_data["max_pillars"]
        self.velos = sorted(glob.glob(os.path.join(root, "velodyne", "*.bin")))
        if not self.velos:
            raise FileNotFoundError(f"No velodyne bins under {root}. See scripts/DATASET.md.")

    def __len__(self) -> int:
        return len(self.velos)

    def __getitem__(self, i: int):
        stem = os.path.splitext(os.path.basename(self.velos[i]))[0]
        pts = np.fromfile(self.velos[i], np.float32).reshape(-1, 4)
        calib = os.path.join(self.root, "calib", stem + ".txt")
        label = os.path.join(self.root, "label_2", stem + ".txt")
        R04, Tr4 = _read_calib(calib)
        gt = _read_labels(label, R04, Tr4) if os.path.exists(label) else np.zeros((0, 7), np.float32)
        pillars, coords, npoints = voxelize(pts, self.pc_range, self.voxel_size,
                                            self.max_points, self.max_pillars)
        return {"pillars": pillars, "coords": coords, "npoints": npoints, "gt": gt}


class SyntheticKITTI3D(Dataset):
    """Point clouds with a handful of box-shaped point clusters as GT."""

    def __init__(self, cfg_data: dict, length: int = 8) -> None:
        self.length = length
        self.pc_range = cfg_data["pc_range"]
        self.voxel_size = cfg_data["voxel_size"]
        self.max_points = cfg_data["max_points"]
        self.max_pillars = cfg_data["max_pillars"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng(i)
        x0, y0, z0, x1, y1, z1 = self.pc_range
        # background ground points
        n_bg = 6000
        bg = np.stack([rng.uniform(x0, x1, n_bg), rng.uniform(y0, y1, n_bg),
                       rng.uniform(z0, z0 + 0.3, n_bg), rng.uniform(0, 1, n_bg)], 1)

        gts, obj_pts = [], []
        for _ in range(rng.integers(1, 4)):
            cx = rng.uniform(x0 + 10, x1 - 10)
            cy = rng.uniform(y0 + 5, y1 - 5)
            cz = z0 + 0.75
            w, l, h = 1.6, 3.9, 1.5
            theta = rng.uniform(-np.pi, np.pi)
            pts = np.stack([
                cx + rng.uniform(-w / 2, w / 2, 300),
                cy + rng.uniform(-l / 2, l / 2, 300),
                cz + rng.uniform(-h / 2, h / 2, 300),
                rng.uniform(0, 1, 300)], 1)
            obj_pts.append(pts)
            gts.append([cx, cy, cz, w, l, h, theta])

        pts = np.concatenate([bg] + obj_pts, 0).astype(np.float32)
        gt = np.array(gts, np.float32).reshape(-1, 7)
        pillars, coords, npoints = voxelize(pts, self.pc_range, self.voxel_size,
                                            self.max_points, self.max_pillars)
        return {"pillars": pillars, "coords": coords, "npoints": npoints, "gt": gt}


def collate(batch):
    return {
        "pillars": torch.from_numpy(np.stack([b["pillars"] for b in batch])),
        "coords": torch.from_numpy(np.stack([b["coords"] for b in batch])),
        "npoints": torch.from_numpy(np.stack([b["npoints"] for b in batch])),
        "gt": [torch.from_numpy(b["gt"]) for b in batch],
    }
