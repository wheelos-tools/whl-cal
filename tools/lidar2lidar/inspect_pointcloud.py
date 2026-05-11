#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.record_adapter import (  # noqa: E402,E501
    Record,
    ensure_record_available,
)
from lidar2lidar.record_utils import discover_record_files  # noqa: E402


def _points_xyz(message) -> np.ndarray:
    if hasattr(message, "points_xyz_array"):
        return np.asarray(message.points_xyz_array(), dtype=np.float64)
    return np.array(
        [(point.x, point.y, point.z) for point in message.point],
        dtype=np.float64,
    )


def _format_xyz(values: np.ndarray) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect PointCloud2 messages from an Apollo record, "
            "including FlatBuffers PCLD payloads."
        )
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="Path to a .record file or split-record directory.",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Point cloud topic to inspect.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=1,
        help="Maximum messages to print.",
    )
    args = parser.parse_args()

    ensure_record_available()
    record_files = discover_record_files(args.record_path)
    inspected = 0
    for record_file in record_files:
        with Record(record_file) as record:
            for topic, message, timestamp_ns in record.read_messages(
                topics=[args.topic]
            ):
                if topic != args.topic:
                    continue
                points = _points_xyz(message)
                finite_mask = (
                    np.isfinite(points).all(axis=1)
                    if points.size
                    else np.empty((0,), dtype=bool)
                )
                finite_points = points[finite_mask] if points.size else points
                frame_id = getattr(
                    getattr(message, "header", None), "frame_id", ""
                ) or getattr(message, "frame_id", "")
                point_count = int(points.shape[0])
                finite_point_count = int(finite_points.shape[0])
                if finite_point_count > 0:
                    xyz_min = finite_points.min(axis=0)
                    xyz_max = finite_points.max(axis=0)
                else:
                    xyz_min = xyz_max = np.zeros(3, dtype=np.float64)
                xyz_min_text = _format_xyz(xyz_min)
                xyz_max_text = _format_xyz(xyz_max)
                print(f"topic={topic}")
                print(f"timestamp_ns={int(timestamp_ns)}")
                print(f"frame_id={frame_id}")
                print(
                    "measurement_time="
                    f"{float(getattr(message, 'measurement_time', 0.0)):.6f}"
                )
                print(f"width={int(getattr(message, 'width', 0))}")
                print(f"height={int(getattr(message, 'height', 0))}")
                print(f"point_count={point_count}")
                print(f"finite_point_count={finite_point_count}")
                print(f"xyz_min={xyz_min_text}")
                print(f"xyz_max={xyz_max_text}")
                print("---")
                inspected += 1
                if inspected >= args.max_messages:
                    return
    raise RuntimeError(f"No point cloud messages found on topic {args.topic}.")


if __name__ == "__main__":
    main()
