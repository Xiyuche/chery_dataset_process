#!/usr/bin/env python3
"""Generate only camera extrinsics in the same numeric ID format as main preprocessing.

Usage:
    python chery_extrinsics_only.py --clip /path/to/clip_xxx --out ./output_dir

It will create (if absent):
    <output_dir>/extrinsics/<cam_id>.txt

Where cam_id follows the mapping used in `chery_preprocess.py`.
Only lidar->camera extrinsics are processed, converted to Waymo coordinate
frame using the same `fix_camera_pose` logic from the full pipeline.
"""
import os
import argparse
import yaml
import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation


def get_camera_mapping() -> Dict[str, int]:
    """Return camera name -> id mapping (mirrors full pipeline)."""
    mapping = {
        "front_wide": 0, "front_main": 1, "left_front": 2, "left_rear": 3,
        "right_front": 4, "right_rear": 5, "rear_main": 6,
        "fisheye_left": 7, "fisheye_rear": 8, "fisheye_front": 9, "fisheye_right": 10
    }
    variants = {k + "_camera": v for k, v in mapping.items()}
    mapping.update(variants)
    return mapping


def read_transform_yaml(yaml_path: str) -> np.ndarray:
    """Read a 4x4 transform matrix from various YAML schema variants.

    Returns identity if file missing.
    """
    if not os.path.exists(yaml_path):
        return np.eye(4)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if data is None:
        return np.eye(4)
    if 'transform' in data:
        tf = data['transform']
        if isinstance(tf, dict) and 'rotation' in tf and 'translation' in tf:
            rot = tf['rotation']
            trans = tf['translation']
            quat = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
            Rm = Rotation.from_quat(quat).as_matrix()
            t = np.array([trans['x'], trans['y'], trans['z']])
            T = np.eye(4)
            T[:3, :3] = Rm
            T[:3, 3] = t
            return T
        else:
            return np.array(tf)
    if 'data' in data:
        arr = np.array(data['data'])
        if arr.size == 16:
            return arr.reshape(4, 4)
    arr = np.array(data)
    if arr.size == 16:
        return arr.reshape(4, 4)
    return np.eye(4)


def fix_camera_pose(T_orig: np.ndarray) -> np.ndarray:
    """Convert original (Z-forward) camera pose to Waymo (X-forward) frame.

    Mirrors implementation in full preprocessing script and returns the inverse
    of adjusted transform to match saved convention.
    """
    T_fix = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    R_orig = T_orig[:3, :3]
    t_orig = T_orig[:3, 3]
    R_new = T_fix @ R_orig
    t_new = T_fix @ t_orig
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return np.linalg.inv(T_new)


def generate_extrinsics_only(clip_dir: str, output_dir: str) -> None:
    """Generate extrinsics/<cam_id>.txt for all cameras with available YAML.

    Args:
        clip_dir: Path to a single Chery clip (contains intrinsics/ & extrinsics/ subdirs)
        output_dir: Destination directory (will create 'extrinsics')
    """
    intrinsics_dir = os.path.join(clip_dir, 'intrinsics')
    lidar2cam_dir = os.path.join(clip_dir, 'extrinsics', 'lidar2camera')
    if not os.path.isdir(intrinsics_dir):
        raise FileNotFoundError(f"Intrinsics dir not found: {intrinsics_dir}")
    if not os.path.isdir(lidar2cam_dir):
        raise FileNotFoundError(f"Extrinsics lidar2camera dir not found: {lidar2cam_dir}")

    os.makedirs(os.path.join(output_dir, 'extrinsics'), exist_ok=True)

    camera_mapping = get_camera_mapping()
    cam_yaml_files = [f for f in os.listdir(intrinsics_dir) if f.endswith('.yaml')]
    processed = 0
    missing = 0
    for cam_yaml in sorted(cam_yaml_files):
        cam_name = cam_yaml[:-5]  # strip .yaml
        if cam_name not in camera_mapping:
            # Accept names like 'front_main_camera' already handled above
            print(f"[Skip] Unknown camera name in mapping: {cam_name}")
            continue
        cam_id = camera_mapping[cam_name]
        # Build extrinsics file name pattern similar to full script
        cam_name_for_extr = cam_name.replace('_', '').replace('camera', '')
        extr_yaml = os.path.join(lidar2cam_dir, f'lidar2{cam_name_for_extr}.yaml')
        if not os.path.exists(extr_yaml):
            print(f"[Warn] Missing extrinsics for {cam_name} -> {extr_yaml}")
            missing += 1
            continue
        T = read_transform_yaml(extr_yaml)
        T_fixed = fix_camera_pose(T)
        out_path = os.path.join(output_dir, 'extrinsics', f'{cam_id}.txt')
        np.savetxt(out_path, T_fixed)
        processed += 1
        print(f"[OK] Saved cam {cam_id} ({cam_name}) extrinsics -> {out_path}")

    print(f"Done. Processed={processed}, Missing={missing}")


def parse_args():
    ap = argparse.ArgumentParser(description='Generate only extrinsics from a Chery clip')
    ap.add_argument('--clip', required=True, help='Path to clip directory (contains intrinsics/, extrinsics/)')
    ap.add_argument('--out', required=True, help='Output directory (will create extrinsics/)')
    return ap.parse_args()


def main():
    # args = parse_args()
    clip_dir = "/home/yuchen/yuchen/repos/cherry-dataset/A车/城市/clip_1717055347001"
    output_dir = "zero/ext_exp/001"
    generate_extrinsics_only(clip_dir, output_dir)


if __name__ == '__main__':
    main()
