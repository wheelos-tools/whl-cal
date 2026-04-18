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
#
# Created Date: 2026-02-09
# Author: daohu527

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lidar2imu.io import load_dataset, write_outputs
from lidar2imu.models import CalibrationConfig
from lidar2imu.pipeline import run_calibration

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LiDAR-to-IMU calibration from curated ground and motion sample features."
    )
    parser.add_argument(
        "--input", required=True, help="Path to a YAML or JSON sample dataset."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2imu/calibration",
        help="Output directory.",
    )
    parser.add_argument(
        "--parent-frame",
        default=None,
        help="Override parent frame, default uses dataset parent_frame.",
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Override child frame, default uses dataset child_frame.",
    )
    parser.add_argument(
        "--loss",
        default=None,
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Robust loss for least-squares stages.",
    )
    parser.add_argument(
        "--min-motion-rotation-deg",
        type=float,
        default=None,
        help="Minimum angular excitation required for motion samples.",
    )
    parser.add_argument(
        "--planar-motion-policy",
        default=None,
        choices=["auto", "free", "freeze_xyyaw"],
        help="How to handle weak planar observability during calibration.",
    )
    args = parser.parse_args()

    dataset, config, raw_payload = load_dataset(
        args.input,
        parent_frame_override=args.parent_frame,
        child_frame_override=args.child_frame,
    )
    config_updates = {}
    if args.loss is not None:
        config_updates["loss"] = args.loss
    if args.min_motion_rotation_deg is not None:
        config_updates["min_motion_rotation_deg"] = args.min_motion_rotation_deg
    if args.planar_motion_policy is not None:
        config_updates["planar_motion_policy"] = args.planar_motion_policy
    if config_updates:
        config = CalibrationConfig(**{**config.__dict__, **config_updates})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running lidar2imu calibration with %d ground samples and %d motion samples.",
        len(dataset.ground_samples),
        len(dataset.motion_samples),
    )
    result = run_calibration(dataset, config=config, output_dir=str(output_dir))
    manifest = write_outputs(
        output_dir=output_dir,
        dataset=dataset,
        initial_transform=result["initial_transform"],
        final_transform=result["final_transform"],
        metrics_output=result["metrics"],
        algorithm_report={
            "input_file": str(Path(args.input).resolve()),
            "config": config.__dict__,
            "dataset_metadata": dataset.metadata,
            "raw_metadata": raw_payload.get("metadata", {}),
            "stages": result["stages"],
        },
        evaluation_report=result["evaluation"],
    )
    logging.info("Saved lidar2imu calibration outputs to %s", output_dir)
    logging.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
