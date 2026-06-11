#!/usr/bin/env python
"""LiDAR->camera extrinsic by point-to-plane bundle adjustment.

Stage 1 (EM match): with the mechanical axis prior R0 and a translation-only
update, predict the board centre in the lidar, crop a tight box there, RANSAC
the board plane. This tracks the *moving* handheld board (a static confounder
cannot follow the camera's vertical trajectory) and yields, per pose, the set
of lidar points on the board.

Stage 2 (bundle): the camera gives a precise board plane (normal n_cam through
centre cc) per pose. Optimise the 6-DoF lidar->camera transform (R,t) so every
lidar board point lands on its camera plane:
    min_{R,t}  sum_ij ( n_cam_i . (R p_ij + t - cc_i) )^2
This uses all board points against the accurate camera plane, so it is robust
to the sparse/noisy per-board lidar normal.
"""
import yaml
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "calibration_data"
OUT = ROOT / "calibration_output"
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # cam->lidar axis map


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


def load_poses(K, D, P, objp):
    poses = []
    for png in sorted(DATA.glob("*.png")):
        n = png.stem
        g = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY)
        f, c = cv2.findChessboardCorners(g, P, FLAGS)
        if not f:
            continue
        cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        ok, rvec, tvec = cv2.solvePnP(objp, c, K, D, flags=cv2.SOLVEPNP_IPPE)
        Rm, _ = cv2.Rodrigues(rvec)
        cc = (Rm @ objp.mean(0).reshape(3, 1) + tvec).flatten()
        n_cam = Rm[:, 2]
        n_cam *= np.sign(-np.dot(n_cam, cc))          # point toward camera
        cloud = np.loadtxt(DATA / f"{n}.pcd", skiprows=11)[:, :3]
        poses.append(dict(name=n, cc=cc, n_cam=n_cam, corners=c.reshape(-1, 2),
                          cloud=cloud))
    return poses


def crop_board(cloud, pL, half):
    m = np.all(np.abs(cloud - pL) < half, axis=1)
    if m.sum() < 25:
        return None
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud[m])
    try:
        _pl, inl = pc.segment_plane(0.035, 3, 3000)
    except RuntimeError:
        return None
    if len(inl) < 20:
        return None
    pts = cloud[m][inl]
    # keep the largest cluster on the plane (drops the holder's body/hands and
    # any coplanar background); the board is one compact blob
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(pts)
    lab = np.array(pc2.cluster_dbscan(eps=0.08, min_points=8))
    if lab.max() >= 0:
        pts = pts[lab == np.bincount(lab[lab >= 0]).argmax()]
    if not (0.25 <= np.linalg.norm(pts.max(0) - pts.min(0)) <= 0.95):
        return None
    return pts


def em_match(poses):
    t = np.zeros(3)
    for it in range(8):
        lcs, ccs = [], []
        for p in poses:
            pts = crop_board(p["cloud"], R0 @ p["cc"] + t, 0.35)
            p["_pts"] = pts
            if pts is not None:
                lcs.append(pts.mean(0)); ccs.append(p["cc"])
        if len(lcs) < 4:
            print(f"  em{it}: {len(lcs)} matches");
            if len(lcs) == 0: break
        lcs, ccs = np.array(lcs), np.array(ccs)
        tn = np.median(lcs - (R0 @ ccs.T).T, axis=0)
        dt = np.linalg.norm(tn - t); t = tn
        res = np.linalg.norm((R0 @ ccs.T).T + t - lcs, axis=1)
        print(f"  em{it}: matches={len(lcs)} med_center_res={np.median(res)*100:.1f}cm dt={dt*100:.1f}cm")
        if dt < 0.003:
            break
    # keep poses whose matched centre tracks the prediction (rejects confounder)
    kept = []
    for p in poses:
        if p["_pts"] is None:
            continue
        r = np.linalg.norm(R0 @ p["cc"] + t - p["_pts"].mean(0))
        p["_res"] = r
        kept.append(p)
    res = np.array([p["_res"] for p in kept])
    thr = max(2 * np.median(res), 0.05)
    kept = [p for p in kept if p["_res"] <= thr]
    print(f"[EM] kept {len(kept)} poses (thr {thr*100:.0f}cm)")
    return kept, t


def residuals(x, kept, w_center=12.0):
    """Point-to-plane (orientation + range) + center-to-center (in-plane
    translation, which fronto-parallel boards leave unconstrained)."""
    rvec, t = x[:3], x[3:]
    R, _ = cv2.Rodrigues(rvec)
    out = []
    for p in kept:
        q = (R @ p["_pts"].T).T + t                    # lidar pts in camera frame
        out.append((q - p["cc"]) @ p["n_cam"])         # signed dist to cam plane
        lc = R @ p["_pts"].mean(0) + t                 # lidar board center in cam
        out.append(w_center * (lc - p["cc"]))          # 3D center anchor
    return np.concatenate(out)


def run():
    K, D, P, objp = load_cfg()
    poses = load_poses(K, D, P, objp)
    print(f"[POSES] {len(poses)} with corners")
    kept, t_em = em_match(poses)
    if len(kept) < 5:
        print("Not enough clean poses; aborting."); return

    # init lidar->camera from inverse of (R0, t_em)
    R_lc0 = R0.T
    rvec0 = cv2.Rodrigues(R_lc0)[0].flatten()
    t_lc0 = -R0.T @ t_em
    x0 = np.concatenate([rvec0, t_lc0])

    r0 = residuals(x0, kept)
    # robust solve, then drop poses with high median point-to-plane and refit
    sol = least_squares(residuals, x0, args=(kept,), method="trf",
                        loss="soft_l1", f_scale=0.02, max_nfev=400)
    pose_med = []
    for p in kept:
        R, _ = cv2.Rodrigues(sol.x[:3])
        d = ((R @ p["_pts"].T).T + sol.x[3:] - p["cc"]) @ p["n_cam"]
        pose_med.append(np.median(np.abs(d)))
    pose_med = np.array(pose_med)
    thr = max(2 * np.median(pose_med), 0.03)
    kept = [p for i, p in enumerate(kept) if pose_med[i] <= thr]
    print(f"[BUNDLE] dropped {int((pose_med > thr).sum())} high-residual poses "
          f"(thr {thr*1000:.0f}mm); refit on {len(kept)}")
    sol = least_squares(residuals, sol.x, args=(kept,), method="trf",
                        loss="soft_l1", f_scale=0.02, max_nfev=400)

    def p2p_only(x):
        R, _ = cv2.Rodrigues(x[:3])
        return np.concatenate([((R @ p["_pts"].T).T + x[3:] - p["cc"]) @ p["n_cam"]
                               for p in kept])
    rf = p2p_only(sol.x)
    print(f"[BUNDLE] point-to-plane RMS: init {np.sqrt((p2p_only(x0)**2).mean())*1000:.1f}mm "
          f"-> final {np.sqrt((rf**2).mean())*1000:.1f}mm  (n_pts={len(rf)})")

    rvec, t = sol.x[:3], sol.x[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    np.set_printoptions(suppress=True, precision=5)
    print("\nExtrinsic (LiDAR -> Camera):\n", np.round(T, 5))

    # board-centre reprojection error
    errs = []
    for p in kept:
        uv, _ = cv2.projectPoints(p["_pts"].mean(0).reshape(1, 3), rvec, t, K, D)
        errs.append(np.linalg.norm(uv.flatten() - p["corners"].mean(0)))
    errs = np.array(errs)
    print(f"board-centre reprojection: median {np.median(errs):.1f}px "
          f"mean {errs.mean():.1f}px (n={len(errs)})")

    Tinv = np.eye(4); Tinv[:3, :3] = R.T; Tinv[:3, 3] = -R.T @ t
    OUT.mkdir(exist_ok=True)
    rep = dict(extrinsic_lidar_to_camera=T.tolist(),
               extrinsic_camera_to_lidar=Tinv.tolist(),
               point_to_plane_rms_mm=float(np.sqrt((rf**2).mean()) * 1000),
               board_center_reproj_px_median=float(np.median(errs)),
               board_center_reproj_px_mean=float(errs.mean()),
               n_poses=len(kept),
               kept_poses=[p["name"] for p in kept])
    yaml.dump(rep, open(OUT / "lidar2camera_extrinsic.yaml", "w"), sort_keys=False)
    print(f"[SAVED] {OUT/'lidar2camera_extrinsic.yaml'}")
    save_overlay(kept, rvec, t, K, D)
    return T


def save_overlay(kept, rvec, t, K, D):
    # pick the pose with the most board points; overlay full cloud + board pts
    p = max(kept, key=lambda q: len(q["_pts"]))
    img = cv2.imread(str(DATA / f"{p['name']}.png"))
    full = np.loadtxt(DATA / f"{p['name']}.pcd", skiprows=11)[:, :3]
    R, _ = cv2.Rodrigues(rvec)
    cam = (R @ full.T).T + t
    front = cam[:, 2] > 0.1
    uv, _ = cv2.projectPoints(full[front], rvec, t, K, D)
    uv = uv.reshape(-1, 2)
    h, w = img.shape[:2]
    for (u, v), z in zip(uv, cam[front, 2]):
        if 0 <= u < w and 0 <= v < h:
            col = (0, int(max(0, 255 - z * 25)), int(min(255, z * 25)))
            cv2.circle(img, (int(u), int(v)), 1, col, -1)
    bp, _ = cv2.projectPoints(p["_pts"], rvec, t, K, D)
    for u, v in bp.reshape(-1, 2):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255), -1)
    cv2.imwrite(str(OUT / "verification_overlay.png"), img)
    print(f"[SAVED] {OUT/'verification_overlay.png'} (pose {p['name']})")


if __name__ == "__main__":
    run()
