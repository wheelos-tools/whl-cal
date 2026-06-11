#!/usr/bin/env python
"""Step 0 - extract a PointCloud2 channel from Apollo Cyber RT bags to PCD.

Runs wherever the bags live (typically on the vehicle / Orin). Needs only the
pure-Python reader:  pip install --user cyber_record protobuf==3.19.4
No Apollo runtime / sourcing required.

Each cloud is written as ASCII PCD (x y z intensity), invalid/zero-return points
dropped, named  <prefix>_<seq>_<lidar_timestamp_ns>.pcd  so the lidar timestamp
is available for camera time-sync downstream.

Usage:
    python3 extract_pcd_from_bag.py \
        --bags '/path/to/bags/all_*' \
        --channel /apollo/sensor/livox/front/PointCloud2 \
        --out /path/to/out_pcd [--prefix livox_front]

Then tar it up:  tar -czf out_pcd.tar.gz -C <parent> out_pcd
"""
import argparse
import glob
import os

import numpy as np
from cyber_record.record import Record

HEADER = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
          "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
          "COUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
          "POINTS {n}\nDATA ascii\n")


def write_pcd(path, xyz, inten):
    with open(path, "w") as fh:
        fh.write(HEADER.format(n=xyz.shape[0]))
        np.savetxt(fh, np.column_stack([xyz, inten]), fmt="%.5f %.5f %.5f %.1f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", required=True,
                    help="glob for bag files, e.g. '/path/all_*'")
    ap.add_argument("--channel", required=True,
                    help="PointCloud2 channel, e.g. /apollo/sensor/livox/front/PointCloud2")
    ap.add_argument("--out", required=True, help="output directory for .pcd")
    ap.add_argument("--prefix", default="cloud")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bags = sorted(glob.glob(args.bags))
    print(f"bags: {len(bags)}  channel: {args.channel}")
    gidx = 0
    for bag in bags:
        cnt = 0
        for _topic, msg, _bt in Record(bag).read_messages(args.channel):
            pts = msg.point
            xyz = np.empty((len(pts), 3), np.float32)
            inten = np.empty(len(pts), np.float32)
            for i, p in enumerate(pts):
                xyz[i] = (p.x, p.y, p.z)
                inten[i] = p.intensity
            keep = np.isfinite(xyz).all(1) & ~(np.abs(xyz) < 1e-6).all(1)
            ts = msg.header.lidar_timestamp
            write_pcd(os.path.join(args.out, f"{args.prefix}_{gidx:05d}_{ts}.pcd"),
                      xyz[keep], inten[keep])
            gidx += 1
            cnt += 1
        print(f"  {os.path.basename(bag)} -> {cnt} clouds (total {gidx})", flush=True)
    print(f"DONE total clouds: {gidx} -> {args.out}")


if __name__ == "__main__":
    main()
