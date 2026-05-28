#!/usr/bin/env python3
"""Summarize whl-cal camera intrinsic runs.

Usage:
  analyze_intrinsic.py <run_dir> [<run_dir> ...]
  analyze_intrinsic.py <parent_dir>      # auto-discovers any subdir with calibration.yaml
  analyze_intrinsic.py --ref-fx 1396 --ref-k1 -0.31 <run_dir>

Prints one summary row per run plus a per-run detail block, with a verdict of
PASS / PASS-soft / FAIL against the thresholds at the top of this file.

The ``--ref-fx`` / ``--ref-k1`` flags compare each fit against the expected
lens-family value; supply your own based on the camera spec sheet or a known-
good prior calibration. Defaults are placeholder values tuned for a generic
~70° HFOV wide-angle IPC on a 1/1.8" sensor at 1920x1080 — override for any
other lens.
"""
import argparse
import glob
import os
import sys

import yaml

# baseline placeholder values; override via --ref-fx / --ref-k1
REF_FX = 1396.0
REF_K1 = -0.31
ROI_TARGET = 0.75
OPT_FX_TARGET = 0.70
BBOX_MEAN_TARGET = 0.08
RMONO_TARGET = 0.5


def find_runs(paths):
    out = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "calibration.yaml")):
            out.append(p)
            continue
        for sub in sorted(glob.glob(os.path.join(p, "*"))):
            if os.path.isfile(os.path.join(sub, "calibration.yaml")):
                out.append(sub)
    return out


def load(run):
    with open(os.path.join(run, "calibration.yaml")) as f:
        y = yaml.safe_load(f)
    K = y["camera_matrix"]["data"]
    D = y["distortion_coefficients"]["data"]
    pvr = y["per_view_reprojection_summary"]
    sq = y["sample_quality"]
    cov = sq["image_coverage"]
    rm = sq["radial_monotonicity"]
    opt = y["undistortion_preview"]["optimized_camera_matrix"]
    roi = y["undistortion_preview"]["valid_roi"]
    iw, ih = y["image_width"], y["image_height"]
    label = os.path.basename(run.rstrip("/"))
    return {
        "label": label,
        "run": run,
        "fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2],
        "D": D,
        "avg_reproj": y["avg_reprojection_error"],
        "pv_mean": pvr["mean"], "pv_p95": pvr["p95"], "pv_max": pvr["max"],
        "N": sq["accepted_sample_count"],
        "cells": cov["occupied_cell_count"],
        "h_span": cov["horizontal_span_ratio"],
        "v_span": cov["vertical_span_ratio"],
        "bbox_mean": cov["bbox_area_ratio"]["mean"],
        "bbox_max": cov["bbox_area_ratio"]["max"],
        "edge_min": cov["edge_margin_px"]["min"],
        "rmono": rm["min_radial_derivative"],
        "opt_fx": opt[0][0],
        "roi_frac": (roi["width"] * roi["height"]) / (iw * ih),
        "roi_wh": (roi["width"], roi["height"]),
        "src": y["capture_runtime"].get("capture_source"),
    }


def verdict(r, ref_fx, ref_k1):
    fx_dev = abs(r["fx"] - ref_fx) / ref_fx
    k1_dev = abs(r["D"][0] - ref_k1) / abs(ref_k1)
    opt_ratio = r["opt_fx"] / r["fx"] if r["fx"] else 0
    reasons = []
    if r["rmono"] <= 0:
        reasons.append(f"rmono={r['rmono']:.2f}(<=0)")
    if fx_dev > 0.20:
        reasons.append(f"fx_dev={fx_dev*100:.0f}%(>20%)")
    if k1_dev > 0.50:
        reasons.append(f"k1_dev={k1_dev*100:.0f}%(>50%)")
    if opt_ratio < OPT_FX_TARGET / 2:
        reasons.append(f"opt_fx_ratio={opt_ratio:.2f}(<0.35)")
    if r["pv_p95"] > 1.0:
        reasons.append(f"pv_p95={r['pv_p95']:.2f}px(>1.0)")
    if reasons:
        return "FAIL", reasons
    soft = []
    if r["rmono"] < RMONO_TARGET:
        soft.append(f"rmono<{RMONO_TARGET}")
    if r["roi_frac"] < ROI_TARGET:
        soft.append(f"roi<{int(ROI_TARGET*100)}%")
    if opt_ratio < OPT_FX_TARGET:
        soft.append(f"opt_fx_ratio<{OPT_FX_TARGET}")
    if r["bbox_mean"] < BBOX_MEAN_TARGET:
        soft.append(f"bbox_mean<{BBOX_MEAN_TARGET}")
    if soft:
        return "PASS-soft", soft
    return "PASS", []


def fmt_metrics(rows):
    hdr = (
        f"{'label':<48}{'N':>3}{'pv_m':>7}{'pv95':>7}{'fx':>8}{'fy':>8}"
        f"{'cx':>7}{'cy':>7}{'bbm':>7}{'bbM':>7}{'edge':>5}{'rmo':>6}"
        f"{'opt%':>6}{'roi%':>6}{'verdict':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r, v in rows:
        verd, _ = v
        print(
            f"{r['label']:<48}{r['N']:>3}"
            f"{r['pv_mean']:>7.3f}{r['pv_p95']:>7.3f}"
            f"{r['fx']:>8.1f}{r['fy']:>8.1f}{r['cx']:>7.1f}{r['cy']:>7.1f}"
            f"{r['bbox_mean']:>7.3f}{r['bbox_max']:>7.3f}{r['edge_min']:>5.0f}"
            f"{r['rmono']:>6.2f}{r['opt_fx']/r['fx']*100:>5.0f}%"
            f"{r['roi_frac']*100:>5.0f}%{verd:>14}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref-fx", type=float, default=REF_FX)
    ap.add_argument("--ref-k1", type=float, default=REF_K1)
    args = ap.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        print("no calibration.yaml under given paths", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run in runs:
        r = load(run)
        v = verdict(r, args.ref_fx, args.ref_k1)
        rows.append((r, v))
    fmt_metrics(rows)

    print()
    print("D = [k1, k2, p1, p2, k3]")
    for r, v in rows:
        verd, reasons = v
        flag = "" if verd == "PASS" else f"  [{verd}]: {', '.join(reasons)}"
        print(
            f"  {r['label']}:"
            f" [{', '.join(f'{x:+.4f}' for x in r['D'])}]"
            f"  roi={r['roi_wh'][0]}x{r['roi_wh'][1]}{flag}"
        )


if __name__ == "__main__":
    main()
