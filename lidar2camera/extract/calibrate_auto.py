#!/usr/bin/env python
"""Calibrate when the camera<->LiDAR relative orientation is unknown.

calibrate_p2plane.py assumes the camera and LiDAR look roughly the same way
(the axis prior R0). That holds for aligned pairs (front cam + front lidar) but
not, e.g., for a side camera paired with a side lidar mounted at a different
yaw -- there the EM prediction misses the board and the solve diverges.

This wrapper searches the relative yaw: for each candidate it sets the prior,
runs the full EM + point-to-plane bundle, and keeps the result with the lowest
board reprojection among physically plausible ones (small camera<->lidar
baseline). The bundle still refines the full 6-DoF rotation, so a yaw-only search
is enough to seed it. Use this when calibrate_p2plane.py diverges or returns an
implausible translation.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import calibrate_p2plane as cp

BASE = cp.R0.copy()
YAWS = list(range(-90, 91, 15))          # relative yaw candidates (deg)
MAX_BASELINE_M = 0.8                      # plausible camera<->lidar separation
MIN_POSES = 8


def Rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def main():
    results = []
    for yaw in YAWS:
        cp.R0 = Rz(yaw) @ BASE
        try:
            T, med, n = cp.run(save=False)
        except Exception:
            continue
        baseline = float(np.linalg.norm(T[:3, 3]))
        ok = n >= MIN_POSES and baseline < MAX_BASELINE_M
        print(f"[AUTO] yaw {yaw:+4d}: reproj {med:5.1f}px  poses {n:2d}  "
              f"|t| {baseline:.2f}m  {'ok' if ok else 'reject'}", flush=True)
        if ok:
            results.append((med, yaw))
    if not results:
        print("[AUTO] no plausible solution; check sensor overlap / data.")
        return
    results.sort()
    best = results[0][1]
    print(f"\n[AUTO] best relative yaw {best:+d} deg "
          f"(reproj {results[0][0]:.1f}px) -> finalizing & saving")
    cp.R0 = Rz(best) @ BASE
    cp.run(save=True)


if __name__ == "__main__":
    main()
