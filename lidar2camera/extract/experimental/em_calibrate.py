#!/usr/bin/env python
"""LiDAR<->camera extrinsic via EM/ICP on board centers (+ normals).

The camera PnP gives a reliable per-pose board center (cc) and plane normal
(n_cam). We seek the rigid transform cam->lidar (R,t):  pL = R*pC + t.
Start from the known axis mapping, then alternate:
  E-step: predict board center in lidar pL=R*cc+t, crop the cloud to a box
          there, RANSAC the board plane -> measured lidar center lc + normal.
  M-step: re-fit (R,t) from the cc<->lc correspondences (Kabsch).
A static confounder object cannot satisfy the moving camera trajectory, so it
is driven out as the prediction walks away from it. Outliers are then dropped
and the fit refined; rotation is finally polished using the plane normals.
"""
import glob
import yaml
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "calibration_data"
PATTERN_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
# axis map cam(x-right,y-down,z-fwd) -> lidar FLU(x-fwd,y-left,z-up)
R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
BOX_HALF = np.array([0.45, 0.45, 0.45])
PLANE_DIST = 0.04


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


def kabsch(A, B):
    """R,t with B ~= R*A + t."""
    a, b = A.mean(0), B.mean(0)
    H = (A - a).T @ (B - b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, b - R @ a


def load_poses(K, D, P, objp):
    poses = []
    for png in sorted(DATA.glob("*.png")):
        n = png.stem
        img = cv2.imread(str(png))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f, c = cv2.findChessboardCorners(g, P, PATTERN_FLAGS)
        if not f:
            continue
        cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        ok, rvec, tvec = cv2.solvePnP(objp, c, K, D, flags=cv2.SOLVEPNP_IPPE)
        Rm, _ = cv2.Rodrigues(rvec)
        cc = (Rm @ objp.mean(0).reshape(3, 1) + tvec).flatten()
        n_cam = Rm[:, 2]                                  # board plane normal
        cloud = np.loadtxt(DATA / f"{n}.pcd", skiprows=11)
        poses.append(dict(name=n, cc=cc, n_cam=n_cam, corners=c,
                          rvec=rvec, tvec=tvec, cloud=cloud))
    return poses


def measure_board(cloud, pL, box_half=0.45):
    xyz = cloud[:, :3]
    m = np.all(np.abs(xyz - pL) < box_half, axis=1)
    if m.sum() < 30:
        return None
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz[m])
    try:
        pl, inl = pc.segment_plane(PLANE_DIST, 3, 400)
    except RuntimeError:
        return None
    if len(inl) < 25:
        return None
    pts = xyz[m][inl]
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    if not (0.25 <= diag <= 1.0):
        return None
    n = np.array(pl[:3]); n /= np.linalg.norm(n)
    return dict(lc=pts.mean(0), n_lid=n, n=len(inl), diag=diag)


def orient(n, toward):
    """Flip normal n so it points toward `toward` direction."""
    return n if np.dot(n, toward) > 0 else -n


def rot_from_normals(Ncam, Nlid):
    """R minimizing ||Nlid - R Ncam|| (unit vectors), via SVD."""
    H = Ncam.T @ Nlid
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return (Vt.T @ np.diag([1, 1, d]) @ U.T).T  # R: Nlid ~= R Ncam


def match_all(poses, R, t, box_half):
    A, B, Ncam, Nlid, names = [], [], [], [], []
    for p in poses:
        pL = R @ p["cc"] + t
        mb = measure_board(p["cloud"], pL, box_half)
        if mb is None:
            continue
        ncam = orient(p["n_cam"], -p["cc"])          # toward camera origin
        nlid = orient(mb["n_lid"], -mb["lc"])        # toward lidar origin
        A.append(p["cc"]); B.append(mb["lc"])
        Ncam.append(ncam); Nlid.append(nlid); names.append(p["name"])
        p["_lc"] = mb["lc"]
    return (np.array(A), np.array(B), np.array(Ncam), np.array(Nlid), names)


def run():
    K, D, P, objp = load_cfg()
    poses = load_poses(K, D, P, objp)
    print(f"[EM] {len(poses)} poses with corners")

    # ---- Phase A: translation-only EM with fixed axis prior R0 ----
    R, t = R0.copy(), np.zeros(3)
    for it in range(8):
        A, B, Ncam, Nlid, names = match_all(poses, R, t, 0.55)
        if len(A) < 4:
            print(f"  A{it}: only {len(A)} matches; stop"); break
        tn = np.median(B - (R0 @ A.T).T, axis=0)       # robust translation
        dt = np.linalg.norm(tn - t); t = tn
        res = np.linalg.norm((R0 @ A.T).T + t - B, axis=1)
        print(f"  A{it}: matches={len(A)} med_res={np.median(res)*100:.1f}cm dt={dt*100:.1f}cm")
        if dt < 0.005:
            break

    # ---- Phase B: rotation from normals, then refine R,t together ----
    for it in range(6):
        A, B, Ncam, Nlid, names = match_all(poses, R, t, 0.40)
        if len(A) < 4:
            print(f"  B{it}: only {len(A)} matches; stop"); break
        R = rot_from_normals(Ncam, Nlid)
        t = np.median(B - (R @ A.T).T, axis=0)
        res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
        nres = np.degrees(np.arccos(np.clip(np.sum(Nlid * (R @ Ncam.T).T, 1), -1, 1)))
        print(f"  B{it}: matches={len(A)} center_med={np.median(res)*100:.1f}cm "
              f"normal_med={np.median(nres):.1f}deg")

    # final correspondences + outlier rejection
    A, B, Ncam, Nlid, names = match_all(poses, R, t, 0.40)
    res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    thr = max(2 * np.median(res), 0.06)
    good = res <= thr
    R = rot_from_normals(Ncam[good], Nlid[good])
    t = np.median(B[good] - (R @ A[good].T).T, axis=0)
    res2 = np.linalg.norm((R @ A[good].T).T + t - B[good], axis=1)
    print(f"\n[FIT] kept {good.sum()}/{len(A)} poses (thr {thr*100:.0f}cm); "
          f"center RMS {np.sqrt((res2**2).mean())*100:.1f}cm")
    print("kept poses:", [names[i] for i in range(len(good)) if good[i]])

    # cam->lidar (R,t)  =>  lidar->camera extrinsic (what we report)
    R_lc = R.T
    t_lc = -R.T @ t
    T = np.eye(4); T[:3, :3] = R_lc; T[:3, 3] = t_lc
    np.set_printoptions(suppress=True, precision=5)
    print("\nExtrinsic (LiDAR -> Camera):\n", np.round(T, 5))

    # reprojection check: project each kept board's lidar points into the image
    errs = []
    for p in poses:
        if "_lc" not in p:
            continue
        # project the board plane center; finer: project lidar board cluster mean
        pc = (R_lc @ p["_lc"].reshape(3, 1) + t_lc.reshape(3, 1)).flatten()
        if pc[2] <= 0:
            continue
        uv, _ = cv2.projectPoints(p["_lc"].reshape(1, 3),
                                  cv2.Rodrigues(R_lc)[0], t_lc, K, D)
        img_center = p["corners"].reshape(-1, 2).mean(0)
        errs.append(np.linalg.norm(uv.flatten() - img_center))
    print(f"board-center reprojection error: median {np.median(errs):.1f}px "
          f"mean {np.mean(errs):.1f}px  (n={len(errs)})")
    return T


if __name__ == "__main__":
    run()
