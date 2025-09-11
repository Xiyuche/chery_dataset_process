#!/usr/bin/env python3
"""
Minimal interactive LiDAR point viewer.

Goals:
- Be tolerant: auto-detect binary/text, float32/float64, and column count.
- Be simple: show just XYZ points (first 3 columns) with constant color by default.
- Not modify any existing project code.

Examples:
    python view_lidar_bin.py /path/to/points.bin
    python view_lidar_bin.py /path/to/points.txt --stride 5
    python view_lidar_bin.py /path/to/000.bin --cols 14 --dtype f32  # override

Tip: If Open3D is missing, install it: pip install open3d
"""

import argparse
import sys
import os
import numpy as np
from typing import Optional, Tuple


def try_load_text(path: str) -> Optional[np.ndarray]:
    """Try loading a text file of numbers into an array (N,M)."""
    try:
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            if arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            else:
                return None
        if arr.shape[1] >= 3:
            return arr.astype(np.float64, copy=False)
    except Exception:
        return None
    return None


def infer_binary_shape(size_bytes: int) -> Optional[Tuple[np.dtype, int]]:
    """Given file size, guess (dtype, cols) for common combos.
    Tries float32/float64 with cols in a priority list.
    """
    candidates_cols = [14, 10, 9, 8, 7, 6, 4, 3]
    for dt in (np.float32, np.float64):
        item = np.dtype(dt).itemsize
        for k in candidates_cols:
            if size_bytes % (item * k) == 0 and size_bytes // (item * k) >= 1:
                return (dt, k)
    return None


def load_points_any(path: str, force_dtype: Optional[str] = None, force_cols: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Load points from binary or text.
    - If force options provided, use them.
    - Else, try binary f32/f64 with common column counts.
    - Else, try text.
    Returns (N, M) float64 array and a short description string.
    """
    # Forced path first
    if force_dtype or force_cols:
        dt = np.float32 if (force_dtype or '').lower() in ['f32', 'float32', '32'] else np.float64
        cols = int(force_cols or 3)
        raw = np.fromfile(path, dtype=dt)
        if raw.size % cols != 0:
            raise ValueError(f"Forced dtype/cols mismatch: file has {raw.size} values, not divisible by {cols}")
        arr = raw.reshape(-1, cols).astype(np.float64)
        return arr, f"binary {dt}x{cols} (forced)"

    # Try binary autodetect
    try:
        size_bytes = os.path.getsize(path)
        guess = infer_binary_shape(size_bytes)
        if guess is not None:
            dt, cols = guess
            raw = np.fromfile(path, dtype=dt)
            arr = raw.reshape(-1, cols).astype(np.float64)
            return arr, f"binary {dt}x{cols} (auto)"
    except Exception:
        pass

    # Try text fallback
    arr = try_load_text(path)
    if arr is not None:
        return arr, "text auto"

    # Final attempt: raw float32/float64 with no reshape if divisible by 3
    for dt in (np.float32, np.float64):
        try:
            raw = np.fromfile(path, dtype=dt)
            if raw.size % 3 == 0:
                return raw.reshape(-1, 3).astype(np.float64), f"binary {dt}x3 (fallback)"
        except Exception:
            continue

    raise ValueError("Could not infer file format (binary or text) with >=3 columns.")


def make_constant_colors(N: int, value: float = 0.85) -> np.ndarray:
    return np.full((N, 3), value, dtype=np.float64)


def downsample_points(pts: np.ndarray, colors: np.ndarray, voxel: float, stride: int, limit: int):
    """Simple downsampling: voxel grid (via rounding) or striding/limiting."""
    if limit and pts.shape[0] > limit:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=limit, replace=False)
        return pts[idx], colors[idx]
    if stride and stride > 1:
        return pts[::stride], colors[::stride]
    if voxel and voxel > 0:
        keys = np.floor(pts / voxel).astype(np.int64)
        # unique rows
        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
        return pts[uniq_idx], colors[uniq_idx]
    return pts, colors


def visualize_open3d(points: np.ndarray, colors: np.ndarray, show_axes: bool = True):
    try:
        import open3d as o3d
    except Exception as e:
        print("Open3D is required for interactive viewing. Install with: pip install open3d", file=sys.stderr)
        raise

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    geoms = [pcd]
    if show_axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    o3d.visualization.draw_geometries(geoms, window_name='LiDAR Viewer',
                                      width=1280, height=800,
                                      point_show_normal=False)


def main():
    parser = argparse.ArgumentParser(description="Simple viewer for LiDAR points (XYZ only by default)")
    parser.add_argument('path', nargs='?', default='/mnt/public/AISIM/yuchen/repos/tools/chery_dataset_process/outputs_unfixed/010/lidar/000.bin',
                        help='Path to points file (.bin or .txt)')
    parser.add_argument('--cols', type=int, default=None, help='Force column count for binary files')
    parser.add_argument('--dtype', choices=['f32', 'f64'], default=None, help='Force dtype for binary files')
    parser.add_argument('--xyz', type=str, default='0,1,2', help='Comma indices for X,Y,Z (e.g., 0,1,2 or 3,4,5)')
    parser.add_argument('--voxel', type=float, default=0.0, help='Voxel downsample size (meters)')
    parser.add_argument('--stride', type=int, default=0, help='Take every k-th point (after load)')
    parser.add_argument('--limit', type=int, default=0, help='Randomly sample at most N points')
    parser.add_argument('--no-axes', action='store_true', help='Hide coordinate axes')

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Load with robust auto-detection or forced settings
    data, desc = load_points_any(args.path, force_dtype=args.dtype, force_cols=args.cols)

    # Parse xyz indices
    try:
        idxs = [int(i.strip()) for i in args.xyz.split(',')]
        if len(idxs) != 3:
            raise ValueError
        if max(idxs) >= data.shape[1]:
            raise IndexError
    except Exception:
        print(f"Invalid --xyz '{args.xyz}'. Falling back to 0,1,2.")
        idxs = [0, 1, 2]

    if data.shape[1] < 3:
        print(f"Data has only {data.shape[1]} columns; need at least 3 for XYZ.", file=sys.stderr)
        sys.exit(2)

    pts = data[:, idxs]
    colors = make_constant_colors(pts.shape[0])

    pts_ds, colors_ds = downsample_points(pts, colors, args.voxel, args.stride, args.limit)

    print(f"Loaded format: {desc}")
    print(f"Points: {pts.shape[0]} -> showing {pts_ds.shape[0]} | XYZ cols: {idxs}")

    visualize_open3d(pts_ds, colors_ds, show_axes=not args.no_axes)


if __name__ == '__main__':
    main()
