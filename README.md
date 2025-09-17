# Chery Dataset Processing Tools

Utilities for preprocessing Chery raw clips into a Waymo‑like structured output and for quick visualization of results (instances, LiDAR, boxes).

## Repository Contents

- `chery_preprocess.py`  Main end‑to‑end clip preprocessing (images, undistortion, dynamic masks, LiDAR merge + transform, ego pose interpolation, instance track export).
- `draw_test.py`         Plotly 3D visualization of one (or multiple) object boxes and camera pose.
- `view_lidar_bin.py`    Lightweight LiDAR point viewer with auto format inference (needs Open3D).
- `vis.ipynb`            (Optional) Notebook for ad‑hoc inspection (not documented here).
- `outputs*/`            Example generated output folders.

## Output Folder Layout
For each processed clip (e.g. `outputs/004/`):
```
images/             000_0.jpg ... (frame_camID)
lidar/              000.bin ... merged multi-LiDAR point clouds (float32 array, 14 cols)
ego_pose/           000.txt ... 4x4 ego (world_from_ego) matrices
intrinsics/         0.txt ... per camera (fx, fy, cx, cy, plus padding)
extrinsics/         0.txt ... lidar->camera (Waymo coord adjusted) 4x4
Dynamic masks:
  dynamic_masks/
    all/            000_0.png (movable objects)
    human/          persons only
    vehicle/        vehicles (non-person movable)
sky_masks/          000_0.png if sky mask available
instances/
  frame_instances.json   {"frame_idx": [instance_ids...]}
  instances_info.json    Per-instance track metadata & per-frame 4x4 transforms
```

LiDAR binary (14 columns) layout (indices):
```
0-2  lidar origin (x,y,z) repeated per point
3-5  point XYZ in main lidar coordinates
6-9  flow (currently zeros)
10   ground label (0)
11   intensity (0)
12   elongation (0)
13   source lidar index (0..3)
```

## Environment Dependencies
Minimal required Python packages:
- numpy
- Pillow
- PyYAML
- tqdm
- scipy
- opencv-python (cv2)
- plotly (for `draw_test.py`)
- open3d (optional; only for `view_lidar_bin.py`)

Install (example):
```
pip install numpy Pillow PyYAML tqdm scipy opencv-python plotly open3d
```
You can omit `plotly` / `open3d` if you do not need those visualizers.

## 1. Preprocess a Single Clip
`chery_preprocess.py` now supports explicit control of worker processes for per‑frame tasks.

Basic (auto workers):
```
python chery_preprocess.py \
  --clip-dir /path/to/clip_1717055347001 \
  --output-dir ./outputs/004
```
Sequential (deterministic, easier debug):
```
python chery_preprocess.py \
  --clip-dir /path/to/clip_1717055347001 \
  --output-dir ./outputs/004 \
  --frame-workers 1
```
Force N workers (e.g. 6):
```
python chery_preprocess.py \
  --clip-dir /path/to/clip_1717055347001 \
  --output-dir ./outputs/004 \
  --frame-workers 6
```
Arguments:
- `--clip-dir` (required/path) Root folder containing `sample_xxxxxxxx` subdirs and `intrinsics/`, `extrinsics/`, `annotation/`, etc.
- `--output-dir` Destination folder (created if missing).
- `--frame-workers` Parallelism for frames: omitted/None = `min(cpu,8)`; 1 = sequential; >=2 = that many processes.

### Multi-Clip Batch (example pattern)
(Commented example exists at bottom of the script.) Pseudocode:
```python
from multiprocessing import Pool, cpu_count
clips = [d for d in os.listdir(root) if d.startswith('clip_')]
with Pool(min(cpu_count(), 16)) as pool:
    pool.starmap(preprocess_chery_clip, [
        (os.path.join(root, c), os.path.join(out_root, c), None)
        for c in clips
    ])
```

## 2. Visualize Instance Boxes & Camera (`draw_test.py`)
Renders selected frame annotations of one instance (or multiple indices) plus a chosen camera transform to an interactive HTML (auto-opens in browser).

Example:
```
python draw_test.py \
  --intrinsics outputs/004/intrinsics/0.txt \
  --extrinsics outputs/004/extrinsics/0.txt \
  --instances outputs/004/instances/instances_info.json \
  --instance-id 3 \
  --indices 0,5,10,20 \
  --axis-len 2.0
```
Key args:
- `--instance-id` String key in `instances_info.json` (after remap they are 0..N-1)
- `--indices` Frame indices within that track to draw (comma-separated)
- `--axis-len` Length of camera axis vectors in meters
Output: `bbox_scene.html` in current directory (opens automatically).

## 3. View LiDAR Points (`view_lidar_bin.py`)
Auto-detects binary/text, float32/float64, and column count. Defaults to first 3 columns as XYZ.

Basic:
```
python view_lidar_bin.py outputs/004/lidar/000.bin
```
Select alternate XYZ columns (e.g. columns 3,4,5 are point coordinates in our 14-col format):
```
python view_lidar_bin.py outputs/004/lidar/000.bin --xyz 3,4,5
```
Downsampling examples:
```
python view_lidar_bin.py outputs/004/lidar/000.bin --stride 5
python view_lidar_bin.py outputs/004/lidar/000.bin --voxel 0.2
python view_lidar_bin.py outputs/004/lidar/000.bin --limit 200000
```
Force format:
```
python view_lidar_bin.py outputs/004/lidar/000.bin --cols 14 --dtype f32 --xyz 3,4,5
```
Hide axes:
```
python view_lidar_bin.py outputs/004/lidar/000.bin --no-axes
```
Requires: `open3d`.

## 4. Dynamic Mask Semantics
Generated under `dynamic_masks/`:
- `all/`    Movable (vehicles + persons + unknown) categories; excludes static cones/barriers
- `human/`  Only `person`
- `vehicle/` Movable excluding `person`
Empty masks still created if no annotations for frame.

## 5. Coordinate Conventions
- Camera extrinsics converted to a Waymo-like frame (X-forward, Y-left, Z-up) via `fix_camera_pose`.
- Ego poses interpolated from `localization.json` timestamps; vertical offset (+1.801m) applied before matrix conversion.
- Object `obj_to_world` matrices in `instances_info.json` are (world_from_object) after combining per-frame object-in-ego with ego pose.

## 6. Troubleshooting
- Missing intrinsics/extrinsics: Script prints warnings; affected cameras skipped.
- No annotation JSON matched: Empty dynamic masks still produced.
- Set `--frame-workers 1` if debugging (ordered, deterministic output & easier prints).
- Large memory usage: Reduce workers or process fewer clips at once.

## 7. Extending
Potential extensions:
- Add SLERP for quaternion pose interpolation.
- Store intensity/elongation if available from raw LiDAR.
- Add per-category dynamic masks beyond (all/human/vehicle).

## 8. License
(Insert your license or usage terms here.)

## 9. Extrinsics Only Quick Tool
If you only need the camera extrinsics (Waymo frame adjusted) without running the full preprocessing pipeline, use:

```
python chery_extrinsics_only.py \
  --clip /path/to/clip_1717055347001 \
  --out ./extrinsics_out
```

This creates:
```
extrinsics_out/
  extrinsics/
    0.txt
    1.txt
    ...
```
Each file is a 4x4 float matrix (lidar->camera) after the same orientation fix as the full pipeline (`fix_camera_pose`). Missing cameras are skipped with a warning. Existing files are overwritten.

---
Questions / improvements welcome.
