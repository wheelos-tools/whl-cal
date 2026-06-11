#!/usr/bin/env python
"""Render the verification overlay on the UNDISTORTED image.

The image is rectified with cv2.undistort (output uses the same K), and the
point cloud is projected with zero distortion so both live in the same pinhole
frame. Uses the saved extrinsic; board crops come from the EM matcher.
"""
import numpy as np
import cv2
import yaml
from pathlib import Path

from calibrate_p2plane import load_cfg, load_poses, em_match, DATA, OUT

ZERO = np.zeros(5)


def main():
    K, D, P, objp = load_cfg()
    rep = yaml.safe_load(open(OUT / "lidar2camera_extrinsic.yaml"))
    T = np.array(rep["extrinsic_lidar_to_camera"])
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    t = T[:3, 3]

    poses = load_poses(K, D, P, objp)
    kept, _ = em_match(poses)
    p = max(kept, key=lambda q: len(q["_pts"]))     # same pose as distorted overlay

    img = cv2.imread(str(DATA / f"{p['name']}.png"))
    und = cv2.undistort(img, K, D)                   # newCameraMatrix defaults to K
    full = np.loadtxt(DATA / f"{p['name']}.pcd", skiprows=11)[:, :3]
    R = T[:3, :3]
    cam = (R @ full.T).T + t
    front = cam[:, 2] > 0.1
    # project with ZERO distortion -> lands in the rectified frame
    uv, _ = cv2.projectPoints(full[front], rvec, t, K, ZERO)
    h, w = und.shape[:2]
    for (u, v), z in zip(uv.reshape(-1, 2), cam[front, 2]):
        if 0 <= u < w and 0 <= v < h:
            col = (0, int(max(0, 255 - z * 25)), int(min(255, z * 25)))
            cv2.circle(und, (int(u), int(v)), 1, col, -1)
    bp, _ = cv2.projectPoints(p["_pts"], rvec, t, K, ZERO)
    for u, v in bp.reshape(-1, 2):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(und, (int(u), int(v)), 3, (0, 0, 255), -1)

    cv2.putText(und, "UNDISTORTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)
    out = OUT / "verification_overlay_undistorted.png"
    cv2.imwrite(str(out), und)
    print(f"[SAVED] {out} (pose {p['name']}, {front.sum()} pts)")


if __name__ == "__main__":
    main()
