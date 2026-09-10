# KITTI 3D Object Detection data

1. Download the KITTI 3D object benchmark from
   <https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>:
   - Velodyne point clouds (`data_object_velodyne.zip`)
   - camera calibration matrices (`data_object_calib.zip`)
   - training labels (`data_object_label_2.zip`)

2. Arrange as:

   ```
   data/kitti_object/training/velodyne/000000.bin
   data/kitti_object/training/calib/000000.txt
   data/kitti_object/training/label_2/000000.txt
   ```

3. Point `configs/kitti.yaml:data.root` at `data/kitti_object/training`.

Boxes are read from `label_2` (camera frame), converted to the LiDAR frame, and
expressed as `[x, y, z, w, l, h, theta]`. The default config trains the **Car**
class; add classes in `dataset.py:CLASS_MAP` and extend `anchor_sizes`.

## Try it with no download

```bash
python scripts/smoke_test.py
python -m pointpillars.train --config configs/smoke.yaml --synthetic
python -m pointpillars.evaluate --config configs/smoke.yaml --synthetic --checkpoint checkpoints/last.pth --iou 0.25
```
