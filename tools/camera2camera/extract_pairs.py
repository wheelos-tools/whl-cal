#!/usr/bin/env python3
"""Extract synchronized PNG pairs from two recorded videos for camera2camera.

The two .mkv files are sampled by wall-clock time (PTS), not by frame number,
so different per-camera fps does not desync the pairs. For each sampled instant
both videos are seeked to that PTS, one frame is read from each, optionally
chessboard-detection-filtered, and saved as pair_NNNN.png with matching stems
under pairs/<pair>/{parent,child}/.

Writes pairs/<pair>/extraction_manifest.yaml capturing per-pair source video,
requested PTS, actual decoded PTS, and the parent-child PTS offset (so you can
see sync drift between the two recordings).

Usage:
  extract_pairs.py <pair_name> <parent.mkv> <child.mkv>
  extract_pairs.py <pair_name> <parent.mkv> <child.mkv> --interval-s 0.5
  extract_pairs.py <pair_name> <parent.mkv> <child.mkv> --no-detect

Output paths are ``pairs/<pair_name>/{parent,child}/pair_NNNN.png`` and
``pairs/<pair_name>/extraction_manifest.yaml`` relative to the script's own
directory.

Resumable: existing pair_NNNN.png in the parent dir is detected and numbering
continues from the next index. Running the script multiple times with the same
``pair_name`` but different video files appends pairs across segments and
records each pair's source video in the manifest.
"""
import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, total, (width, height)


def seek_and_read(cap, t_ms, skip_after_seek=2):
    """Seek to t_ms; cv2 snaps to nearest keyframe. Skip a couple of frames
    to avoid post-seek decode garbage, then return the first clean one."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
    for _ in range(skip_after_seek):
        cap.grab()
    ok, frame = cap.read()
    actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    return (frame if ok else None), actual_ms


def detect_chessboard_fast(gray, pattern):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ok, _ = cv2.findChessboardCorners(gray, pattern, flags)
    return ok


def detect_chessboard_with_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    return ok, corners


def mean_corner_displacement(corners_a, corners_b):
    """Mean L2 displacement in pixels between two corner sets of equal length."""
    a = corners_a.reshape(-1, 2)
    b = corners_b.reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())


def next_pair_index(out_parent):
    if not os.path.isdir(out_parent):
        return 1
    existing = sorted(
        f for f in os.listdir(out_parent)
        if f.startswith("pair_") and f.endswith(".png")
    )
    if not existing:
        return 1
    last = existing[-1]
    return int(last[len("pair_"):-len(".png")]) + 1


def load_existing_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("pairs", []) or []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pair", help="pair name (e.g. 64_66) — selects pairs/<pair>/ output dirs")
    ap.add_argument("parent_video", help="path to parent (closer-to-cam64) .mkv")
    ap.add_argument("child_video", help="path to child .mkv")
    ap.add_argument("--interval-s", type=float, default=1.0,
                    help="sampling interval in seconds (default 1.0)")
    ap.add_argument("--pattern", default="11,8",
                    help="chessboard inner corners as W,H (default 11,8)")
    ap.add_argument("--no-detect", action="store_true",
                    help="save every sampled pair without chessboard detection")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="cap on saved pairs (0 = unlimited)")
    ap.add_argument("--stability-window-ms", type=float, default=300.0,
                    help="check board stability over this lookback window per camera (default 300)")
    ap.add_argument("--stability-max-px", type=float, default=4.0,
                    help="max mean corner displacement (px) over the window to accept (default 4.0)")
    ap.add_argument("--no-stability", action="store_true",
                    help="disable the stability check (board can be moving)")
    args = ap.parse_args()

    pw, ph = (int(x) for x in args.pattern.split(","))
    pattern = (pw, ph)

    pair_root = os.path.join(SCRIPT_DIR, "pairs", args.pair)
    out_parent = os.path.join(pair_root, "parent")
    out_child = os.path.join(pair_root, "child")
    os.makedirs(out_parent, exist_ok=True)
    os.makedirs(out_child, exist_ok=True)

    pcap, p_fps, p_total, p_size = open_video(args.parent_video)
    ccap, c_fps, c_total, c_size = open_video(args.child_video)
    p_dur = p_total / p_fps if p_fps > 0 else 0
    c_dur = c_total / c_fps if c_fps > 0 else 0
    overlap = min(p_dur, c_dur)

    print(f"parent: {os.path.basename(args.parent_video)}  "
          f"{p_size[0]}x{p_size[1]}  {p_total} frames @ {p_fps:.2f} fps  ({p_dur:.1f}s)")
    print(f"child:  {os.path.basename(args.child_video)}  "
          f"{c_size[0]}x{c_size[1]}  {c_total} frames @ {c_fps:.2f} fps  ({c_dur:.1f}s)")
    print(f"sampling every {args.interval_s:.2f}s for {overlap:.1f}s of overlap")
    print(f"chessboard filter: {'OFF' if args.no_detect else f'{pw}x{ph}'}")
    if args.no_detect and not args.no_stability:
        print("  (note: stability check requires chessboard detection; disabling stability check)")
        args.no_stability = True
    print(f"stability filter:  "
          f"{'OFF' if args.no_stability else f'<= {args.stability_max_px:.1f} px corner motion over {args.stability_window_ms:.0f} ms'}")

    if p_size != c_size:
        print(f"  ! warning: parent {p_size} and child {c_size} differ — "
              "c2c assumes per-camera intrinsics already encode that.")

    next_idx = next_pair_index(out_parent)
    manifest_path = os.path.join(pair_root, "extraction_manifest.yaml")
    existing_pairs = load_existing_manifest(manifest_path)
    if existing_pairs:
        print(f"resuming: {len(existing_pairs)} pair(s) already in manifest; "
              f"next index pair_{next_idx:04d}")

    parent_abs = os.path.abspath(args.parent_video)
    child_abs = os.path.abspath(args.child_video)
    # Preserve any source-video paths from prior runs so the manifest shows
    # every (parent, child) pair the manifest was built from.
    prior_sources = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            prev = yaml.safe_load(f) or {}
        prior_sources = prev.get("source_videos", []) or []
    sources_seen = list(prior_sources)
    this_source = {"parent_video": parent_abs, "child_video": child_abs}
    if this_source not in sources_seen:
        sources_seen.append(this_source)

    manifest = {
        "pair": args.pair,
        "source_videos": sources_seen,
        "parent_resolution": list(p_size),
        "child_resolution": list(c_size),
        "parent_fps": p_fps,
        "child_fps": c_fps,
        "pattern_size": [pw, ph],
        "interval_s": args.interval_s,
        "detect_chessboard": not args.no_detect,
        "last_extracted_at": datetime.now().isoformat(timespec="seconds"),
        "pairs": list(existing_pairs),
    }

    saved = 0
    inspected = 0
    skipped_no_board = 0
    skipped_motion = 0
    skipped_read_fail = 0
    t_ms = 0.0
    interval_ms = args.interval_s * 1000.0

    try:
        while t_ms < overlap * 1000.0:
            inspected += 1
            p_frame, p_pts = seek_and_read(pcap, t_ms)
            c_frame, c_pts = seek_and_read(ccap, t_ms)
            if p_frame is None or c_frame is None:
                skipped_read_fail += 1
                t_ms += interval_ms
                continue
            p_corners = c_corners = None
            if not args.no_detect:
                p_gray = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
                c_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
                if args.no_stability:
                    p_ok = detect_chessboard_fast(p_gray, pattern)
                    c_ok = detect_chessboard_fast(c_gray, pattern)
                else:
                    p_ok, p_corners = detect_chessboard_with_corners(p_gray, pattern)
                    c_ok, c_corners = detect_chessboard_with_corners(c_gray, pattern)
                if not (p_ok and c_ok):
                    skipped_no_board += 1
                    print(f"  t={t_ms/1000:6.2f}s  skip (board parent={int(p_ok)} child={int(c_ok)})")
                    t_ms += interval_ms
                    continue
            p_motion_px = c_motion_px = None
            if not args.no_stability:
                t_back_ms = max(0.0, t_ms - args.stability_window_ms)
                p_prev, _ = seek_and_read(pcap, t_back_ms)
                c_prev, _ = seek_and_read(ccap, t_back_ms)
                if p_prev is None or c_prev is None:
                    skipped_read_fail += 1
                    t_ms += interval_ms
                    continue
                p_prev_ok, p_prev_corners = detect_chessboard_with_corners(
                    cv2.cvtColor(p_prev, cv2.COLOR_BGR2GRAY), pattern)
                c_prev_ok, c_prev_corners = detect_chessboard_with_corners(
                    cv2.cvtColor(c_prev, cv2.COLOR_BGR2GRAY), pattern)
                if not (p_prev_ok and c_prev_ok):
                    skipped_no_board += 1
                    print(f"  t={t_ms/1000:6.2f}s  skip (board missing in lookback frame)")
                    t_ms += interval_ms
                    continue
                p_motion_px = mean_corner_displacement(p_prev_corners, p_corners)
                c_motion_px = mean_corner_displacement(c_prev_corners, c_corners)
                if max(p_motion_px, c_motion_px) > args.stability_max_px:
                    skipped_motion += 1
                    print(f"  t={t_ms/1000:6.2f}s  skip motion "
                          f"(parent={p_motion_px:.1f}px child={c_motion_px:.1f}px, "
                          f"limit {args.stability_max_px:.1f})")
                    t_ms += interval_ms
                    continue
            stem = f"pair_{next_idx:04d}.png"
            p_path = os.path.join(out_parent, stem)
            c_path = os.path.join(out_child, stem)
            cv2.imwrite(p_path, p_frame)
            cv2.imwrite(c_path, c_frame)
            motion_note = ""
            if p_motion_px is not None:
                motion_note = f" motion(p={p_motion_px:.1f}px c={c_motion_px:.1f}px)"
            print(f"  t={t_ms/1000:6.2f}s  saved {stem}  "
                  f"(p_pts={p_pts:.0f}ms c_pts={c_pts:.0f}ms "
                  f"offset={p_pts-c_pts:+.0f}ms){motion_note}")
            entry = {
                "stem": stem,
                "parent_video": parent_abs,
                "child_video": child_abs,
                "requested_t_ms": float(t_ms),
                "parent_pts_ms": float(p_pts),
                "child_pts_ms": float(c_pts),
                "pts_offset_ms": float(p_pts - c_pts),
                "detected_chessboard": not args.no_detect,
            }
            if p_motion_px is not None:
                entry["stability_window_ms"] = float(args.stability_window_ms)
                entry["parent_motion_px"] = float(p_motion_px)
                entry["child_motion_px"] = float(c_motion_px)
            manifest["pairs"].append(entry)
            next_idx += 1
            saved += 1
            if args.max_pairs and saved >= args.max_pairs:
                break
            t_ms += interval_ms
    finally:
        pcap.release()
        ccap.release()
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)

    print()
    print(f"done. inspected={inspected}  saved={saved}  "
          f"skipped_no_board={skipped_no_board}  "
          f"skipped_motion={skipped_motion}  skipped_read_fail={skipped_read_fail}")
    print(f"out parent: {out_parent}")
    print(f"out child:  {out_child}")
    print(f"manifest:   {manifest_path}")
    if saved == 0:
        print("! no pairs saved. Try --no-detect to dump every sampled pair and inspect them manually.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
