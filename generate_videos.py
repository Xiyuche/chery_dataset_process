#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate per-camera mp4 videos from a flat image folder.

Image filename pattern expected: <frame>_<camera>.<ext>
Examples: 000_0.jpg, 000_10.jpg, 195_7.png

Usage (example):
  python generate_videos.py \
      --image-dir outputs_unfixed_lidarfixed_ego/013/images \
      --out-dir outputs_unfixed_lidarfixed_ego/013/videos \
      --fps 10

Dependencies: opencv-python (cv2). Install if missing:
  pip install opencv-python
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

try:
    import cv2  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit("请先安装 opencv-python: pip install opencv-python") from e

FILENAME_RE = re.compile(r"^(?P<frame>\d+)_(?P<cam>\d+)\.(?P<ext>jpg|jpeg|png|bmp)$", re.IGNORECASE)


@dataclass
class ImageEntry:
    frame: int
    cam: int
    path: Path


def scan_images(image_dir: Path) -> List[ImageEntry]:
    entries: List[ImageEntry] = []
    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        frame = int(m.group("frame"))
        cam = int(m.group("cam"))
        entries.append(ImageEntry(frame=frame, cam=cam, path=p))
    if not entries:
        raise ValueError(f"目录中未找到匹配模式的图片: {image_dir}")
    return entries


def group_by_camera(entries: Iterable[ImageEntry]) -> Dict[int, List[ImageEntry]]:
    cams: Dict[int, List[ImageEntry]] = {}
    for e in entries:
        cams.setdefault(e.cam, []).append(e)
    for cam_entries in cams.values():
        cam_entries.sort(key=lambda x: x.frame)
    return cams


def detect_resolution(first_image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(first_image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {first_image_path}")
    h, w = img.shape[:2]
    return w, h


def write_video(cam: int, entries: List[ImageEntry], out_dir: Path, fps: float, fourcc: str = "mp4v", keep_missing_warn: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = detect_resolution(entries[0].path)
    video_path = out_dir / f"cam{cam}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {video_path}")

    # Check frame continuity (optional warning)
    expected = entries[0].frame
    for e in entries:
        # Fill gaps? Currently just warn.
        if e.frame != expected and keep_missing_warn:
            print(f"[警告] 相机 {cam} 缺失帧: 期望 {expected:06d} 实际 {e.frame:06d}")
            expected = e.frame  # resync
        img = cv2.imread(str(e.path))
        if img is None:
            print(f"[警告] 读取失败跳过: {e.path}")
            continue
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
        expected += 1
    writer.release()
    print(f"生成视频: {video_path} (帧数 {len(entries)})")
    return video_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="按相机编号把帧序列打包为 mp4 视频")
    ap.add_argument("--image-dir", required=True, help="图片所在目录 (包含 frame_cam.jpg)")
    ap.add_argument("--out-dir", default="videos", help="输出视频目录")
    ap.add_argument("--fps", type=float, default=10.0, help="帧率 (默认 10)")
    ap.add_argument("--fourcc", default="mp4v", help="视频编码 fourcc (默认 mp4v, 可试 H264/avc1) ")
    ap.add_argument("--cams", nargs="*", type=int, help="只处理指定相机 id (空则全部)")
    ap.add_argument("--skip-missing-warn", action="store_true", help="不输出缺帧警告")
    return ap.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    entries = scan_images(image_dir)
    cams = group_by_camera(entries)
    target_cams = args.cams if args.cams else sorted(cams.keys())
    print(f"发现相机: {sorted(cams.keys())}; 将处理: {target_cams}")
    for cam in target_cams:
        if cam not in cams:
            print(f"[跳过] 未找到相机 {cam} 的图片")
            continue
        write_video(cam, cams[cam], out_dir, fps=args.fps, fourcc=args.fourcc, keep_missing_warn=not args.skip_missing_warn)
    print("全部完成")


if __name__ == "__main__":  # pragma: no cover
    main()
