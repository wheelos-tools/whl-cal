#!/usr/bin/env python3
"""Summarize whl-cal camera2camera (c2c) extrinsic runs.

Usage:
  analyze.py <run_dir> [<run_dir> ...]
  analyze.py <parent_dir>     # auto-discovers any subdir with metrics.yaml

For each run, parses metrics.yaml + diagnostics/data_quality.yaml and prints:
  - pair counts (paired / accepted / acceptance ratio)
  - final RMS, per-pair p95, holdout p95, epipolar p95
  - translation magnitude, t_xyz, Euler RPY (degrees)
  - delta vs initial guess
  - coverage cells (parent, child)
  - pose diversity (depth span, tilt span)
  - LOO repeatability (translation std, rotation std)
  - verdict: PASS / PASS-soft / FAIL with the failing gates

Verdict thresholds match the c2c quickstart and methodology docs.
"""
import argparse
import glob
import math
import os
import sys

import yaml

# acceptance gates (from camera2camera_quickstart.md)
FINAL_RMS_PX_MAX = 1.0
PAIR_RMS_P95_PX_MAX = 1.5
HOLDOUT_RMS_P95_PX_MAX = 1.5
EPIPOLAR_P95_PX_MAX = 1.0
ACCEPTED_PAIR_RATIO_MIN = 0.5
MIN_OCCUPIED_CELLS = 4  # of 9 in the 3x3 image grid
MIN_DEPTH_SPAN_M = 0.30
MIN_TILT_SPAN_DEG = 30.0
LOO_TRANSLATION_STD_MAX_M = 0.02
LOO_ROTATION_STD_MAX_DEG = 1.0


def find_runs(paths):
    out = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "metrics.yaml")):
            out.append(p)
            continue
        for sub in sorted(glob.glob(os.path.join(p, "*"))):
            if os.path.isfile(os.path.join(sub, "metrics.yaml")):
                out.append(sub)
    return out


def loo_stats(diag_dir):
    """Mean/std translation_norm_m + rotation_deg across LOO trials, parsed from
    leave_one_out_trials.csv (faster than re-parsing the metrics.yaml giant)."""
    path = os.path.join(diag_dir, "leave_one_out_trials.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            cells = line.strip().split(",")
            if len(cells) < len(header):
                continue
            rows.append(dict(zip(header, cells)))
    if not rows:
        return None
    cols = list(rows[0].keys())
    t_col = next((c for c in cols if "translation_norm" in c), None)
    r_col = next((c for c in cols if "rotation_deg" in c), None)
    if not (t_col and r_col):
        return None
    t = [float(r[t_col]) for r in rows if r.get(t_col) not in (None, "")]
    r = [float(row[r_col]) for row in rows if row.get(r_col) not in (None, "")]
    if not (t and r):
        return None

    def stats(v):
        n = len(v)
        m = sum(v) / n
        s = math.sqrt(sum((x - m) ** 2 for x in v) / n) if n > 0 else 0.0
        return m, s, max(v)

    t_mean, t_std, t_max = stats(t)
    r_mean, r_std, r_max = stats(r)
    return {
        "trials": len(rows),
        "t_mean_m": t_mean, "t_std_m": t_std, "t_max_m": t_max,
        "r_mean_deg": r_mean, "r_std_deg": r_std, "r_max_deg": r_max,
    }


def quat_to_euler_deg(qx, qy, qz, qw):
    """ZYX-Tait-Bryan (yaw-pitch-roll) extraction in degrees."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    deg = 180.0 / math.pi
    return roll * deg, pitch * deg, yaw * deg


def load(run):
    with open(os.path.join(run, "metrics.yaml")) as f:
        m = yaml.safe_load(f)
    with open(os.path.join(run, "calibrated_tf.yaml")) as f:
        tf = yaml.safe_load(f)
    summ = m["summary"]
    coarse = m["coarse_metrics"]
    accept = m["final_acceptance"]
    t = tf["transform"]["translation"]
    q = tf["transform"]["rotation"]
    roll, pitch, yaw = quat_to_euler_deg(q["x"], q["y"], q["z"], q["w"])
    t_xyz = (t["x"], t["y"], t["z"])
    t_norm = math.sqrt(sum(c * c for c in t_xyz))
    diag_dir = os.path.join(run, "diagnostics")
    loo = loo_stats(diag_dir)
    pose_div = coarse.get("pose_diversity", {}) or {}
    return {
        "label": os.path.basename(run.rstrip("/")),
        "run": run,
        "parent_frame": tf["header"].get("frame_id"),
        "child_frame": tf.get("child_frame_id"),
        "pair_count": int(summ.get("pair_count", 0)),
        "paired_count": int(_find_gate(accept, "paired_pair_count", "paired_count", default=0)),
        "accepted_ratio": float(coarse.get("accepted_pair_ratio", 0.0)),
        "initial_rms": float(summ.get("initial_rms_px", 0.0)),
        "final_rms": float(summ.get("final_rms_px", 0.0)),
        "pair_rms_p95": float(coarse.get("pair_reprojection_rms_px", {}).get("p95", 0.0)),
        "pair_rms_max": float(coarse.get("pair_reprojection_rms_px", {}).get("max", 0.0)),
        "epipolar_p95": float(coarse.get("epipolar_error_px", {}).get("p95", 0.0)),
        "holdout_p95": float(coarse.get("holdout_reprojection_rms_px", {}).get("p95", 0.0)),
        "t_xyz": t_xyz,
        "t_norm": t_norm,
        "rpy": (roll, pitch, yaw),
        "delta_init_t_m": float(summ.get("delta_to_initial", {}).get("translation_norm_m", 0.0)),
        "delta_init_rot_deg": float(summ.get("delta_to_initial", {}).get("rotation_deg", 0.0)),
        "parent_cells": int(coarse.get("parent_image_coverage", {}).get("occupied_cell_count", 0)),
        "child_cells": int(coarse.get("child_image_coverage", {}).get("occupied_cell_count", 0)),
        "depth_span_m": float(pose_div.get("depth_span_m", 0.0)),
        "tilt_span_deg": float(pose_div.get("tilt_span_deg", 0.0)),
        "loo": loo,
        "release_ready": bool(accept.get("release_ready", False)),
        "acceptance_status": accept.get("status", "?"),
    }


def _find_gate(accept, name, field, default=None):
    for g in accept.get("gates", []) or []:
        if g.get("name") == name:
            ev = g.get("evidence", "") or ""
            for tok in ev.replace(",", " ").split():
                if tok.startswith(f"{field}="):
                    try:
                        return float(tok.split("=", 1)[1])
                    except ValueError:
                        pass
    return default


def verdict(r):
    fails = []  # hard fails
    soft = []   # warnings

    # core fit
    if r["final_rms"] > FINAL_RMS_PX_MAX:
        fails.append(f"final_rms={r['final_rms']:.2f}>{FINAL_RMS_PX_MAX}")
    if r["pair_rms_p95"] > PAIR_RMS_P95_PX_MAX:
        soft.append(f"pair_p95={r['pair_rms_p95']:.2f}>{PAIR_RMS_P95_PX_MAX}")
    if r["holdout_p95"] > HOLDOUT_RMS_P95_PX_MAX:
        soft.append(f"holdout_p95={r['holdout_p95']:.2f}>{HOLDOUT_RMS_P95_PX_MAX}")
    if r["epipolar_p95"] > EPIPOLAR_P95_PX_MAX:
        soft.append(f"epi_p95={r['epipolar_p95']:.2f}>{EPIPOLAR_P95_PX_MAX}")

    # data sufficiency (these often reveal "low RMS but globally wrong")
    if r["accepted_ratio"] < ACCEPTED_PAIR_RATIO_MIN:
        soft.append(f"accept_ratio={r['accepted_ratio']:.2f}<{ACCEPTED_PAIR_RATIO_MIN}")
    if r["parent_cells"] < MIN_OCCUPIED_CELLS:
        soft.append(f"parent_cells={r['parent_cells']}<{MIN_OCCUPIED_CELLS}")
    if r["child_cells"] < MIN_OCCUPIED_CELLS:
        soft.append(f"child_cells={r['child_cells']}<{MIN_OCCUPIED_CELLS}")
    if r["depth_span_m"] < MIN_DEPTH_SPAN_M:
        soft.append(f"depth_span={r['depth_span_m']:.2f}m<{MIN_DEPTH_SPAN_M}")
    if r["tilt_span_deg"] < MIN_TILT_SPAN_DEG:
        soft.append(f"tilt_span={r['tilt_span_deg']:.1f}deg<{MIN_TILT_SPAN_DEG}")

    # repeatability
    if r["loo"]:
        if r["loo"]["t_std_m"] > LOO_TRANSLATION_STD_MAX_M:
            soft.append(f"loo_t_std={r['loo']['t_std_m']*1000:.1f}mm>{LOO_TRANSLATION_STD_MAX_M*1000:.0f}mm")
        if r["loo"]["r_std_deg"] > LOO_ROTATION_STD_MAX_DEG:
            soft.append(f"loo_r_std={r['loo']['r_std_deg']:.2f}deg>{LOO_ROTATION_STD_MAX_DEG}")

    if fails:
        return "FAIL", fails + soft
    if soft:
        return "PASS-soft", soft
    return "PASS", []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        print("no metrics.yaml under given paths", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run in runs:
        try:
            r = load(run)
        except Exception as e:
            print(f"  skip {run}: {e}", file=sys.stderr)
            continue
        rows.append((r, verdict(r)))

    # Compact metric table
    hdr = (
        f"{'pair':<7}{'frames':>7}{'N':>4}{'acc%':>6}"
        f"{'rms':>6}{'p_p95':>7}{'epi95':>7}{'ho95':>7}"
        f"{'|T|m':>7}{'cellsP':>7}{'cellsC':>7}"
        f"{'depΔm':>7}{'tiltΔ':>7}{'verdict':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r, v in rows:
        verd, _ = v
        label_short = (r["parent_frame"] or "").replace("camera_", "") + "_" + (r["child_frame"] or "").replace("camera_", "")
        print(
            f"{label_short:<7}{r['paired_count']:>7.0f}{r['pair_count']:>4}{r['accepted_ratio']*100:>5.0f}%"
            f"{r['final_rms']:>6.2f}{r['pair_rms_p95']:>7.2f}{r['epipolar_p95']:>7.2f}{r['holdout_p95']:>7.2f}"
            f"{r['t_norm']:>7.3f}{r['parent_cells']:>7d}{r['child_cells']:>7d}"
            f"{r['depth_span_m']:>7.3f}{r['tilt_span_deg']:>7.1f}{verd:>14}"
        )

    # Per-run detail
    print()
    for r, v in rows:
        verd, reasons = v
        roll, pitch, yaw = r["rpy"]
        tx, ty, tz = r["t_xyz"]
        print(f"== {r['parent_frame']} -> {r['child_frame']}  ({r['label']})")
        print(f"   translation_m  ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})  |T|={r['t_norm']:.3f}")
        print(f"   rpy_deg        roll={roll:+.2f}  pitch={pitch:+.2f}  yaw={yaw:+.2f}")
        print(f"   rms_px         init={r['initial_rms']:.3f}  final={r['final_rms']:.3f}"
              f"  pair_p95={r['pair_rms_p95']:.3f}  pair_max={r['pair_rms_max']:.3f}"
              f"  epi_p95={r['epipolar_p95']:.3f}  holdout_p95={r['holdout_p95']:.3f}")
        print(f"   delta vs init  {r['delta_init_t_m']*1000:.1f}mm  {r['delta_init_rot_deg']:.2f}deg")
        print(f"   pairs          paired={r['paired_count']:.0f}  accepted={r['pair_count']}  ratio={r['accepted_ratio']*100:.0f}%")
        print(f"   coverage       parent_cells={r['parent_cells']}/9  child_cells={r['child_cells']}/9"
              f"  depth_span={r['depth_span_m']*100:.1f}cm  tilt_span={r['tilt_span_deg']:.1f}deg")
        if r["loo"]:
            l = r["loo"]
            print(f"   LOO ({l['trials']})    t_std={l['t_std_m']*1000:.1f}mm  t_max={l['t_max_m']*1000:.1f}mm"
                  f"  r_std={l['r_std_deg']:.2f}deg  r_max={l['r_max_deg']:.2f}deg")
        print(f"   release_ready  {r['release_ready']}  (status={r['acceptance_status']})  verdict={verd}")
        if reasons:
            print(f"   issues         " + "; ".join(reasons))
        print()


if __name__ == "__main__":
    main()
