#!/usr/bin/env python3
"""
Stitch per-frame LiDAR PLYs into a single world-aligned PLY using ego poses.

Features:
- Two-pass streaming writer (low memory): first counts vertices, then writes.
- Reads ASCII PLY with XYZ only; ignores extra properties by selecting first 3 cols.
- Applies 4x4 ego pose transform per frame (assumed world_from_ego).
- Binary little-endian PLY output by default (much smaller than ASCII).

Usage examples:
    python stitch_lidar_with_ego.py \
        /mnt/public/AISIM/yuchen/repos/tools/chery_dataset_process/outputs_unfixed_lidarfixed_mirror/013/lidar_debug \
        --output merged_world.ply --binary

    # Downsample by taking every 3rd point to reduce size
    python stitch_lidar_with_ego.py <lidar_debug_dir> --stride 3

Notes:
- Expects sibling ego_pose directory: <scene_dir>/ego_pose/NNN.txt
- Will match frames by 3-digit stem in filenames (e.g., 012.ply -> 012.txt).
"""

import argparse
import os
import re
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm


PLY_FLOAT32_LE = '<f4'


def parse_ply_header(path: str) -> Tuple[int, int, str]:
    """Parse ASCII PLY header and return (vertex_count, header_lines, format).
    format is the string after 'format ', e.g., 'ascii 1.0'.
    """
    vertex_count = None
    header_lines = 0
    ply_format = None
    with open(path, 'r') as f:
        first = f.readline().strip()
        if first != 'ply':
            raise ValueError(f"Not a PLY file: {path}")
        header_lines = 1
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith('format '):
                ply_format = s[len('format '):]
            if s.startswith('element vertex'):
                try:
                    vertex_count = int(s.split()[-1])
                except Exception:
                    raise ValueError(f"Bad vertex count line in {path}: {s}")
            if s == 'end_header':
                break
        else:
            raise ValueError(f"No end_header in {path}")
    if vertex_count is None:
        raise ValueError(f"No vertex element in {path}")
    if ply_format is None:
        raise ValueError(f"No format line in {path}")
    if not ply_format.startswith('ascii'):
        raise ValueError(f"Only ASCII PLY is supported for input, got '{ply_format}' in {path}")
    return vertex_count, header_lines, ply_format


def load_xyz_from_ascii_ply(path: str, header_lines: int, n: int, stride: int = 1) -> np.ndarray:
    """Load first 3 columns (XYZ) from ASCII PLY efficiently."""
    # numpy.loadtxt is fast enough; usecols ensures we take only first 3 columns.
    # max_rows supports Python 3.8+.
    if stride < 1:
        stride = 1
    data = np.loadtxt(path, dtype=np.float64, comments=None, skiprows=header_lines, usecols=(0, 1, 2), max_rows=n)
    if stride > 1:
        data = data[::stride]
    return data


def load_pose_4x4(path: str) -> np.ndarray:
    """Load 4x4 pose matrix from text file (4 lines, 4 numbers each)."""
    mat = np.loadtxt(path, dtype=np.float64)
    mat = mat.reshape(4, 4)
    return mat


def list_frame_stems(lidar_dir: str) -> List[str]:
    stems: List[str] = []
    for name in sorted(os.listdir(lidar_dir)):
        if not name.lower().endswith('.ply'):
            continue
        stem = os.path.splitext(name)[0]
        if re.fullmatch(r'\d{3}', stem):
            stems.append(stem)
    return stems


def write_ply_binary_f32(output_path: str, all_xyz_iter, total_count: int) -> None:
    """Write binary little-endian PLY with only XYZ float32 vertices.
    all_xyz_iter should yield chunks of shape (M, 3) float64/float32.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {total_count}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'end_header\n'
    ).encode('ascii')
    with open(output_path, 'wb') as f:
        f.write(header)
        for xyz in all_xyz_iter:
            arr = np.asarray(xyz, dtype=PLY_FLOAT32_LE)
            f.write(arr.tobytes())


def write_ply_ascii(output_path: str, all_xyz_iter, total_count: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {total_count}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for xyz in all_xyz_iter:
            np.savetxt(f, xyz, fmt='%.6f %.6f %.6f')


def main():
    ap = argparse.ArgumentParser(description='Stitch LiDAR PLY frames into a world PLY using ego poses')
    ap.add_argument('lidar_debug_dir', help='Path to lidar_debug directory containing NNN.ply files')
    ap.add_argument('--ego-dir', default=None, help='Path to ego_pose directory (defaults to sibling of lidar_debug)')
    ap.add_argument('--output', default='merged_world.ply', help='Output PLY filename (placed next to lidar_debug by default)')
    # Output format flags
    fmt_group = ap.add_mutually_exclusive_group()
    fmt_group.add_argument('--ascii', action='store_true', help='Write ASCII PLY')
    fmt_group.add_argument('--binary', action='store_true', help='Write binary PLY (default)')
    ap.add_argument('--stride', type=int, default=1, help='Optional downsample stride (take every k-th point)')
    args = ap.parse_args()

    lidar_dir = os.path.abspath(args.lidar_debug_dir)
    if not os.path.isdir(lidar_dir):
        raise SystemExit(f"Not a directory: {lidar_dir}")

    scene_dir = os.path.dirname(lidar_dir)
    ego_dir = os.path.abspath(args.ego_dir) if args.ego_dir else os.path.join(scene_dir, 'ego_pose')
    if not os.path.isdir(ego_dir):
        raise SystemExit(f"ego_pose directory not found: {ego_dir}")

    stems = list_frame_stems(lidar_dir)
    if not stems:
        raise SystemExit(f"No NNN.ply files in {lidar_dir}")

    # Pass 1: count total vertices (respecting stride) and check pose availability
    total = 0
    infos: List[Tuple[str, str, int, int]] = []  # (ply_path, pose_path, n_vertices, header_lines)
    for stem in stems:
        ply_path = os.path.join(lidar_dir, f'{stem}.ply')
        pose_path = os.path.join(ego_dir, f'{stem}.txt')
        if not os.path.isfile(pose_path):
            # Skip if pose missing
            continue
        n_vert, header_lines, fmt = parse_ply_header(ply_path)
        used = n_vert // max(1, args.stride)
        total += used
        infos.append((ply_path, pose_path, n_vert, header_lines))

    if not infos:
        raise SystemExit("No matching frame pairs (PLY+pose) found.")

    # Determine output path
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(lidar_dir, out_path)

    # Prepare iterator for second pass (streaming)
    def xyz_iter():
        for ply_path, pose_path, n_vert, header_lines in tqdm(infos, desc='Merging', unit='frame'):
            pose = load_pose_4x4(pose_path)
            # Assume pose is world_from_ego. Transform ego-frame points to world.
            R = pose[:3, :3]
            t = pose[:3, 3]
            pts = load_xyz_from_ascii_ply(ply_path, header_lines, n_vert, stride=args.stride)
            # world = R @ pts.T + t -> (3,N)
            world = (R @ pts.T).T + t
            yield world

    # Write output
    # Default is binary unless --ascii was provided
    if args.ascii and not args.binary:
        write_ply_ascii(out_path, xyz_iter(), total)
    else:
        write_ply_binary_f32(out_path, xyz_iter(), total)

    print(f"Wrote {total} points to: {out_path}")


if __name__ == '__main__':
    main()
