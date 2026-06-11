#!/usr/bin/env python
"""Crop each pose's LiDAR cloud down to just the calibration board.

For every pose we know, from the camera, the board's *bearing* in the camera
frame (via solvePnP on the 2D corners). Cameras face forward and the cam<->lidar
baseline is small, so that bearing maps reliably into the LiDAR frame even though
the range carries an unknown baseline offset. We therefore:
  1. crop to the forward region,
  2. RANSAC-segment planar clusters, keep board-sized ones,
  3. among those inside the PnP bearing cone, pick the most checkerboard-like
     (highest LiDAR-intensity spread / stripe alternation),
  4. overwrite NNNN.pcd with the board points so reference_based.py's single
     dominant-plane segmentation lands on the board.

Run with --apply to overwrite; default is a dry-run report.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "calibration_data"
FULL = ROOT / "calibration_data_full"     # backup of uncropped clouds

BOARD_DIAG = np.hypot(0.405, 0.540)
DIAG_LO, DIAG_HI = 0.45, 1.15
NPTS_LO, NPTS_HI = 120, 4000
BEARING_CONE_DEG = 18.0
CROP = dict(x=(0.6, 6.0), y=(-5.0, 5.0), z=(-1.5, 2.0))


def load_cfg():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    K = np.array(cfg["camera"]["intrinsics"], float)
    D = np.array(cfg["camera"]["distortion"], float)
    P = tuple(cfg["checkerboard"]["pattern_size"])
    sq = float(cfg["checkerboard"]["square_size"])
    objp = np.zeros((P[0] * P[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:P[0], 0:P[1]].T.reshape(-1, 2)
    objp *= sq
    return K, D, P, objp


def load_xyzi(p):
    return np.loadtxt(p, skiprows=11)


def write_pcd(path, d):
    n = d.shape[0]
    with open(path, "w") as fh:
        fh.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                 "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
                 "COUNT 1 1 1 1\n"
                 f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                 f"POINTS {n}\nDATA ascii\n")
        np.savetxt(fh, d, fmt="%.5f %.5f %.5f %.1f")


def pnp_bearing(name, K, D, P, objp):
    img = cv2.imread(str(DATA / f"{name}.png"))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f, c = cv2.findChessboardCorners(
        g, P, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not f:
        return None
    cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    ok, rvec, tvec = cv2.solvePnP(objp, c, K, D, flags=cv2.SOLVEPNP_IPPE)
    Rm, _ = cv2.Rodrigues(rvec)
    cc = (Rm @ objp.mean(0).reshape(3, 1) + tvec).flatten()   # cam frame
    predL = np.array([cc[2], -cc[0], -cc[1]])                 # rough lidar FLU
    az = np.degrees(np.arctan2(predL[1], predL[0]))
    el = np.degrees(np.arctan2(predL[2], np.hypot(predL[0], predL[1])))
    return predL, az, el


def istd_score(pts, it):
    return float(it.std())


def find_board(d, az, el):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    m = ((x > CROP["x"][0]) & (x < CROP["x"][1]) &
         (y > CROP["y"][0]) & (y < CROP["y"][1]) &
         (z > CROP["z"][0]) & (z < CROP["z"][1]) & np.isfinite(x))
    sub = d[m]
    if len(sub) < NPTS_LO:
        return None
    xyz = sub[:, :3].copy()
    inten = sub[:, 3].copy()
    remaining = np.arange(len(xyz))
    best = None
    for _ in range(10):
        if len(remaining) < NPTS_LO:
            break
        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(xyz[remaining])
        try:
            _pl, inl = sp.segment_plane(0.02, 3, 600)
        except RuntimeError:
            break
        if len(inl) < 40:
            break
        gi = remaining[inl]
        pp = xyz[gi]
        # largest euclidean cluster of this plane
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pp)
        lab = np.array(pc.cluster_dbscan(eps=0.10, min_points=10))
        if lab.max() >= 0:
            keep = lab == np.bincount(lab[lab >= 0]).argmax()
            pp = pp[keep]
            gi = gi[keep]
        diag = np.linalg.norm(pp.max(0) - pp.min(0))
        cen = pp.mean(0)
        c_az = np.degrees(np.arctan2(cen[1], cen[0]))
        c_el = np.degrees(np.arctan2(cen[2], np.hypot(cen[0], cen[1])))
        daz = abs(c_az - az)
        del_ = abs(c_el - el)
        if (DIAG_LO <= diag <= DIAG_HI and NPTS_LO <= len(pp) <= NPTS_HI
                and daz <= BEARING_CONE_DEG and del_ <= BEARING_CONE_DEG):
            sc = istd_score(pp, inten[gi])
            cand = dict(pts=sub[gi], cen=cen, diag=diag, n=len(pp),
                        daz=daz, del_=del_, score=sc)
            if best is None or sc > best["score"]:
                best = cand
        mask = np.ones(len(remaining), bool)
        mask[inl] = False
        remaining = remaining[mask]
    return best


def main(apply):
    K, D, P, objp = load_cfg()
    names = sorted(p.stem for p in DATA.glob("*.pcd"))
    if apply and not FULL.exists():
        FULL.mkdir()
        for n in names:
            (FULL / f"{n}.pcd").write_bytes((DATA / f"{n}.pcd").read_bytes())
        print(f"[BACKUP] full clouds -> {FULL}")
    results, dropped = [], []
    for n in names:
        br = pnp_bearing(n, K, D, P, objp)
        if br is None:
            print(f"{n}: no corners"); dropped.append(n); continue
        predL, az, el = br
        d = load_xyzi(FULL / f"{n}.pcd" if (FULL / f"{n}.pcd").exists()
                      else DATA / f"{n}.pcd")
        b = find_board(d, az, el)
        if b is None:
            print(f"{n}: NO board in bearing cone (az={az:.0f} el={el:.0f})")
            dropped.append(n); continue
        c = b["cen"]
        print(f"{n}: board n={b['n']:4d} diag={b['diag']:.2f} "
              f"cen=({c[0]:.2f},{c[1]:+.2f},{c[2]:+.2f}) "
              f"daz={b['daz']:.1f} del={b['del_']:.1f} istd={b['score']:.1f}")
        results.append((n, b))
        if apply:
            write_pcd(DATA / f"{n}.pcd", b["pts"])
    if apply:
        # drop poses with no confident board so calibration isn't polluted;
        # remaining png/pcd names stay paired under sorted glob.
        for n in dropped:
            for ext in (".pcd", ".png"):
                f = DATA / f"{n}{ext}"
                if f.exists():
                    f.unlink()
        print(f"\n[APPLY] cropped {len(results)} clouds to board region; "
              f"dropped {len(dropped)} poses without a confident board")
    return results


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
