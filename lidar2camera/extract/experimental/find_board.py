#!/usr/bin/env python
"""Locate the calibration board plane inside a cluttered LiDAR cloud.

Strategy: crop to the plausible forward region, then iteratively RANSAC-segment
planes; for each plane, take its largest Euclidean cluster and score how
"board-like" it is (bbox diagonal ~ board diagonal, moderate point count, plane
not gigantic). Returns the inlier points of the best candidate.

Board: 8x11 interior corners, 0.045 m square -> 9x12 squares = 0.405 x 0.540 m,
full diagonal ~0.67 m (a bit more with the white border).
"""
import sys
import numpy as np
import open3d as o3d

BOARD_DIAG = np.hypot(0.405, 0.540)        # ~0.675 m
DIAG_LO, DIAG_HI = 0.45, 1.10              # accept bbox diagonal in this band
NPTS_LO, NPTS_HI = 120, 6000

# forward crop where a hand-held board ~2-5 m ahead can live (lidar FLU frame)
CROP = dict(x=(1.0, 6.0), y=(-4.5, 4.5), z=(-1.5, 2.0))


def load_xyzi(path):
    return np.loadtxt(path, skiprows=11)


def crop(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    m = ((x > CROP["x"][0]) & (x < CROP["x"][1]) &
         (y > CROP["y"][0]) & (y < CROP["y"][1]) &
         (z > CROP["z"][0]) & (z < CROP["z"][1]))
    return d[m]


def largest_cluster(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    lab = np.array(pc.cluster_dbscan(eps=0.10, min_points=10))
    if lab.max() < 0:
        return pts
    best = np.bincount(lab[lab >= 0]).argmax()
    return pts[lab == best]


def find_board(d, verbose=False):
    sub = crop(d)
    if len(sub) < NPTS_LO:
        return None, None
    xyz = sub[:, :3].copy()
    inten = sub[:, 3].copy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    candidates = []
    remaining = np.arange(len(xyz))
    for it in range(8):
        if len(remaining) < NPTS_LO:
            break
        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(xyz[remaining])
        try:
            plane, inl = sp.segment_plane(0.02, 3, 800)
        except RuntimeError:
            break
        if len(inl) < 30:
            break
        glob_inl = remaining[inl]
        plane_pts = xyz[glob_inl]
        clu = largest_cluster(plane_pts)
        diag = np.linalg.norm(clu.max(0) - clu.min(0))
        normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        vert = abs(normal[2])           # ~0 => vertical plane (board faces sideways)
        if verbose:
            print(f"  it{it}: plane inl={len(inl)} cluster={len(clu)} "
                  f"diag={diag:.2f} vert={vert:.2f}")
        if DIAG_LO <= diag <= DIAG_HI and NPTS_LO <= len(clu) <= NPTS_HI:
            # recover intensity of the clustered inlier points
            from scipy.spatial import cKDTree
            tree = cKDTree(sub[:, :3])
            _, idx = tree.query(clu, k=1)
            clu_i = inten[idx]
            istd = float(clu_i.std())
            # board-likeness: checker pattern -> high intensity spread; plus
            # bbox diagonal close to the true board diagonal
            score = istd - 8.0 * abs(diag - BOARD_DIAG)
            candidates.append((score, clu, istd, diag))
        # remove this plane's inliers and continue searching
        mask = np.ones(len(remaining), bool)
        mask[inl] = False
        remaining = remaining[mask]
    if not candidates:
        return None, sub
    candidates.sort(key=lambda c: c[0], reverse=True)
    if verbose:
        for sc, clu, istd, diag in candidates:
            print(f"   cand score={sc:.1f} istd={istd:.1f} diag={diag:.2f} n={len(clu)}")
    return candidates[0][1], sub


if __name__ == "__main__":
    import glob
    for p in sorted(glob.glob("lidar2camera/calibration_data/*.pcd")):
        d = load_xyzi(p)
        board, _ = find_board(d, verbose=False)
        if board is None:
            print(f"{p.split('/')[-1]}: NO board found")
        else:
            c = board.mean(0)
            diag = np.linalg.norm(board.max(0) - board.min(0))
            print(f"{p.split('/')[-1]}: board {len(board)} pts  "
                  f"center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  diag={diag:.2f}m")
