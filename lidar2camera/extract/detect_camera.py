#!/usr/bin/env python
"""Step 1: detect checkerboard in the camera video, map to clk_unix_ns.

Reads the mkv with OpenCV (frame index == frames.csv frame_idx), samples every
detect_stride-th frame, runs the same corner finder the calibrator uses, and
writes a JSON list of candidate poses (frame_idx, clk_unix_ns, corner centroid,
span) plus the full-res PNG for every detection. All paths/params come from
../config.yaml (the `data:` and `checkerboard:` sections).
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


MKV = _path(CFG["data"]["camera_video"])
CSV = _path(CFG["data"]["camera_csv"])
PATTERN = tuple(CFG["checkerboard"]["pattern_size"])     # interior corners
STRIDE = int(CFG["data"].get("detect_stride", 3))
MAX_FRAME = int(CFG["data"].get("max_frame", 0)) or None  # 0 -> no cap
OUT = Path(__file__).resolve().parent / "cam_candidates"
OUT.mkdir(exist_ok=True)


def load_clk(csv_path):
    clk = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            clk[int(row["frame_idx"])] = int(row["clk_unix_ns"])
    return clk


def find_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return corners


def main():
    clk = load_clk(CSV)
    cap = cv2.VideoCapture(str(MKV))
    idx = -1
    cands = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if MAX_FRAME is not None and idx > MAX_FRAME:
            break
        if idx % STRIDE != 0:
            continue
        if idx not in clk:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)
        if corners is None:
            continue
        c = corners.reshape(-1, 2)
        cands.append({
            "frame_idx": idx,
            "clk_unix_ns": clk[idx],
            "centroid": [float(c[:, 0].mean()), float(c[:, 1].mean())],
            "span_px": float(np.linalg.norm(c.max(0) - c.min(0))),
        })
        # stash frame so we don't re-decode later
        cv2.imwrite(str(OUT / f"frame_{idx:06d}.png"), frame)
        if len(cands) % 20 == 0:
            print(f"  ...{len(cands)} detections (scanned to frame {idx})", flush=True)
    cap.release()
    print(f"[CAM] checkerboard detected in {len(cands)} sampled frames", flush=True)
    with open(OUT / "candidates.json", "w") as fh:
        json.dump(cands, fh, indent=2)
    print(f"[CAM] wrote {OUT/'candidates.json'}", flush=True)


if __name__ == "__main__":
    main()
