#!/usr/bin/env python
"""Stage 2: select diverse board poses from camera detections, then pull the
nearest-in-time hesai cloud for each and write paired NNNN.png / NNNN.pcd.

Sync key: camera clk_unix_ns  <->  lidar header.lidar_timestamp (both NTP unix ns).
"""
import csv
import glob
import json
import shutil
from pathlib import Path

import numpy as np
from cyber_record.record import Record

HERE = Path(__file__).resolve().parent
CAND = HERE / "cam_candidates" / "candidates.json"
BAGS = sorted(glob.glob(
    "/Users/jie/Desktop/extrinsic_bags_141128_00000-00002/all_*"))
LIDAR_CH = "/apollo/sensor/hesai/main_front/PointCloud2"
OUT = HERE.parent / "calibration_data"     # what config.yaml points at

# pose selection: group consecutive detections that sit at the same board
# placement (centroid close + small time gap), keep the sharpest of each group.
GROUP_GAP_NS = int(0.8e9)
GROUP_MOVE_PX = 90.0
MAX_MATCH_NS = int(0.06e9)   # accept lidar cloud within 60 ms of the image
MIN_POINTS = 200             # drop empty/padding-only clouds


def select_poses(cands):
    cands = sorted(cands, key=lambda d: d["clk_unix_ns"])
    groups, cur = [], []
    for c in cands:
        if not cur:
            cur = [c]
            continue
        last = cur[-1]
        dt = c["clk_unix_ns"] - last["clk_unix_ns"]
        move = np.hypot(c["centroid"][0] - last["centroid"][0],
                        c["centroid"][1] - last["centroid"][1])
        if dt <= GROUP_GAP_NS and move <= GROUP_MOVE_PX:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    if cur:
        groups.append(cur)
    # one representative per placement: the largest (closest/most frontal) board
    reps = [max(g, key=lambda d: d["span_px"]) for g in groups]
    print(f"[SELECT] {len(cands)} detections -> {len(groups)} placements")
    return reps


def write_pcd(path, xyz, intensity):
    n = xyz.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA ascii\n")
    with open(path, "w") as fh:
        fh.write(header)
        buf = np.column_stack([xyz, intensity])
        np.savetxt(fh, buf, fmt="%.5f %.5f %.5f %.1f")


def msg_to_xyz(msg):
    pts = msg.point
    xyz = np.empty((len(pts), 3), np.float32)
    inten = np.empty(len(pts), np.float32)
    for i, p in enumerate(pts):
        xyz[i] = (p.x, p.y, p.z)
        inten[i] = p.intensity
    finite = np.isfinite(xyz).all(1)
    nonzero = ~(np.abs(xyz) < 1e-6).all(1)
    keep = finite & nonzero
    return xyz[keep], inten[keep]


def main():
    cands = json.load(open(CAND))
    reps = select_poses(cands)
    targets = sorted(reps, key=lambda d: d["clk_unix_ns"])
    tgt_ns = np.array([t["clk_unix_ns"] for t in targets], dtype=np.int64)
    best = [None] * len(targets)   # (abs_dt, bag, msg) per target

    print(f"[LIDAR] scanning {len(BAGS)} bags for {len(targets)} targets...")
    for bag in BAGS:
        r = Record(bag)
        for _topic, msg, _t in r.read_messages(LIDAR_CH):
            lts = msg.header.lidar_timestamp
            i = int(np.argmin(np.abs(tgt_ns - lts)))
            dt = abs(int(tgt_ns[i]) - int(lts))
            if dt <= MAX_MATCH_NS and (best[i] is None or dt < best[i][0]):
                best[i] = (dt, bag, msg)
        print(f"  scanned {Path(bag).name}", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = []
    out_idx = 0
    for i, t in enumerate(targets):
        if best[i] is None:
            print(f"[SKIP] pose t={t['clk_unix_ns']} frame={t['frame_idx']}: "
                  f"no lidar within {MAX_MATCH_NS/1e6:.0f} ms")
            continue
        dt, _bag, msg = best[i]
        xyz, inten = msg_to_xyz(msg)
        if xyz.shape[0] < MIN_POINTS:
            print(f"[SKIP] pose frame={t['frame_idx']}: only {xyz.shape[0]} pts")
            continue
        name = f"{out_idx:04d}"
        src_png = HERE / "cam_candidates" / f"frame_{t['frame_idx']:06d}.png"
        shutil.copy(src_png, OUT / f"{name}.png")
        write_pcd(OUT / f"{name}.pcd", xyz, inten)
        manifest.append({
            "name": name, "frame_idx": t["frame_idx"],
            "cam_clk_ns": t["clk_unix_ns"],
            "lidar_ts_ns": int(msg.header.lidar_timestamp),
            "sync_dt_ms": dt / 1e6, "n_points": int(xyz.shape[0]),
        })
        print(f"[PAIR] {name}: frame {t['frame_idx']}  dt={dt/1e6:.1f}ms  "
              f"pts={xyz.shape[0]}")
        out_idx += 1

    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print(f"\n[DONE] wrote {out_idx} pose pairs to {OUT}")


if __name__ == "__main__":
    main()
