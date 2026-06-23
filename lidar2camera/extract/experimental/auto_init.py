"""Derive the camera->lidar rotation/translation prior from data, with no axis
assumption. For each pose collect *several* board-sized planar candidates in the
LiDAR; RANSAC over (camera board centre <-> candidate) correspondences finds the
single rigid transform that the real board obeys across poses. Clutter (walls,
mounted boards, signs) cannot satisfy the moving camera trajectory, so it drops
out. Returns R0 (cam->lidar) and t init for the bundle.
"""
import numpy as np
import open3d as o3d

FWD = dict(x=(0.8, 6.0), y=(-4.5, 4.5), z=(-1.3, 1.6))


def board_candidates(cloud, max_k=6):
    """Up to max_k board-sized planar cluster centres in the forward region."""
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    m = ((x > FWD["x"][0]) & (x < FWD["x"][1]) & (y > FWD["y"][0]) &
         (y < FWD["y"][1]) & (z > FWD["z"][0]) & (z < FWD["z"][1]))
    pts = cloud[m, :3]
    it = cloud[m, 3] if cloud.shape[1] > 3 else None
    if len(pts) < 50:
        return []
    rem = np.arange(len(pts))
    cands = []
    for _ in range(12):
        if len(rem) < 50:
            break
        sp = o3d.geometry.PointCloud()
        sp.points = o3d.utility.Vector3dVector(pts[rem])
        try:
            _pl, inl = sp.segment_plane(0.03, 3, 300)
        except RuntimeError:
            break
        if len(inl) < 25:
            break
        gi = rem[inl]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts[gi])
        lab = np.array(pc.cluster_dbscan(eps=0.12, min_points=8))
        for lb in set(lab[lab >= 0]):
            cl = pts[gi][lab == lb]
            diag = np.linalg.norm(cl.max(0) - cl.min(0))
            if 0.4 <= diag <= 1.15 and 40 <= len(cl) <= 6000:
                score = float(it[gi][lab == lb].std()) if it is not None else len(cl)
                cands.append((cl.mean(0), score))
        mask = np.ones(len(rem), bool)
        mask[inl] = False
        rem = rem[mask]
    cands.sort(key=lambda c: -c[1])           # prefer high intensity spread
    return [c[0] for c in cands[:max_k]]


def kabsch(A, B):
    a, b = A.mean(0), B.mean(0)
    H = (A - a).T @ (B - b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, b - R @ a


def auto_init(poses, tol=0.08, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    cc = np.array([p["cc"] for p in poses])
    cand = [board_candidates(p["cloud"]) for p in poses]
    have = [i for i, c in enumerate(cand) if c]
    if len(have) < 4:
        return None
    best = None
    for _ in range(iters):
        s = rng.choice(have, 3, replace=False)
        A = cc[s]
        B = np.array([cand[i][rng.integers(len(cand[i]))] for i in s])
        R, t = kabsch(A, B)
        inl, resid = [], 0.0
        for i in have:
            pred = R @ cc[i] + t
            ds = [np.linalg.norm(pred - c) for c in cand[i]]
            j = int(np.argmin(ds))
            if ds[j] < tol:
                inl.append((i, j)); resid += ds[j]
        if best is None or (len(inl), -resid) > (len(best[0]), -best[1]):
            best = (inl, resid, R, t)
    inl, _, R, t = best
    A = cc[[i for i, _ in inl]]
    B = np.array([cand[i][j] for i, j in inl])
    R, t = kabsch(A, B)
    res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    return dict(R=R, t=t, n_inliers=len(inl), rms_cm=float(np.sqrt((res**2).mean()) * 100))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "lidar2camera/extract")
    from calibrate_p2plane import load_cfg, load_poses, DATA
    K, D, P, objp = load_cfg()
    poses = load_poses(K, D, P, objp)
    for p in poses:                                  # reload with intensity
        p["cloud"] = np.loadtxt(DATA / f"{p['name']}.pcd", skiprows=11)
    r = auto_init(poses)
    np.set_printoptions(suppress=True, precision=4)
    if r:
        print(f"inliers {r['n_inliers']}  centre RMS {r['rms_cm']:.1f}cm")
        print("R0 (cam->lidar):\n", np.round(r["R"], 4))
        print("t:", np.round(r["t"], 3))
    else:
        print("auto_init failed")
