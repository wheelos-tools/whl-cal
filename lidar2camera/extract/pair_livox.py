#!/usr/bin/env python
"""Step 2: pair camera board poses with pre-extracted lidar PCDs by timestamp.

PCDs are named <prefix>_<seq>_<lidar_ts_ns>.pcd (see extract_pcd_from_bag.py).
Camera detections carry clk_unix_ns. Both are NTP unix-ns, so we select diverse
stationary poses and match each to the nearest cloud in time, writing paired
NNNN.png / NNNN.pcd into calibration_data. Paths/params come from ../config.yaml.
"""
import glob
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


CAND = HERE / "cam_candidates" / "candidates.json"
LIVOX = sorted(glob.glob(str(_path(CFG["data"]["livox_pcd_dir"]) / "*.pcd")))
OUT = _path(CFG.get("data_directory", "calibration_data"))

MAX_MATCH_NS = int(CFG["data"].get("sync_tol_ms", 60) * 1e6)
N_POSES = int(CFG["data"].get("n_poses", 24))   # over-select; calibrator filters
STATIONARY_VEL = 50.0          # px/s; avoid motion smear + sync error


def select_poses(cands):
    """Stationary + diverse: farthest-point sampling over (u, v, span)."""
    cands = sorted(cands, key=lambda d: d["clk_unix_ns"])
    cen = np.array([d["centroid"] for d in cands])
    t = np.array([d["clk_unix_ns"] for d in cands], float) / 1e9
    mot = np.r_[0, np.hypot(np.diff(cen[:, 0]), np.diff(cen[:, 1]))]
    dt = np.r_[1, np.diff(t)]
    vel = mot / np.maximum(dt, 1e-3)
    idx = np.where(vel < STATIONARY_VEL)[0]
    if len(idx) < N_POSES:
        idx = np.arange(len(cands))
    # normalized feature space (image position + depth proxy = span)
    feat = np.column_stack([cen[idx, 0] / 1920, cen[idx, 1] / 1080,
                            np.array([cands[i]["span_px"] for i in idx]) / 784])
    sel = [int(np.argmax(feat[:, 2]))]          # start with the closest board
    d2 = np.sum((feat - feat[sel[0]]) ** 2, axis=1)
    while len(sel) < min(N_POSES, len(idx)):
        nxt = int(np.argmax(d2))
        sel.append(nxt)
        d2 = np.minimum(d2, np.sum((feat - feat[nxt]) ** 2, axis=1))
    reps = [cands[idx[s]] for s in sel]
    print(f"[SELECT] {len(cands)} detections, {len(idx)} stationary "
          f"-> {len(reps)} diverse poses")
    return sorted(reps, key=lambda d: d["clk_unix_ns"])


def main():
    cands = json.load(open(CAND))
    reps = select_poses(cands)
    lts = np.array([int(re.search(r"_(\d+)\.pcd$", f).group(1)) for f in LIVOX],
                   dtype=np.int64)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest, out_idx = [], 0
    for t in reps:
        clk = t["clk_unix_ns"]
        j = int(np.argmin(np.abs(lts - clk)))
        dt = abs(int(lts[j]) - clk)
        if dt > MAX_MATCH_NS:
            print(f"[SKIP] frame {t['frame_idx']}: nearest livox {dt/1e6:.0f} ms")
            continue
        name = f"{out_idx:04d}"
        shutil.copy(HERE / "cam_candidates" / f"frame_{t['frame_idx']:06d}.png",
                    OUT / f"{name}.png")
        shutil.copy(LIVOX[j], OUT / f"{name}.pcd")
        manifest.append({"name": name, "frame_idx": t["frame_idx"],
                         "cam_clk_ns": clk, "livox_ts_ns": int(lts[j]),
                         "sync_dt_ms": dt / 1e6,
                         "livox_file": Path(LIVOX[j]).name})
        print(f"[PAIR] {name}: frame {t['frame_idx']} dt={dt/1e6:.1f}ms "
              f"<- {Path(LIVOX[j]).name}")
        out_idx += 1
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print(f"\n[DONE] wrote {out_idx} pose pairs to {OUT}")


if __name__ == "__main__":
    main()
