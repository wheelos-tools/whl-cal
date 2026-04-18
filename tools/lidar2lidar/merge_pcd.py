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

"""Merge a source PCD into a target PCD using a provided transform."""

from __future__ import annotations

import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.cli import load_initial_transform, load_point_clouds, save_registered_point_cloud


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a source PCD into a target PCD using a JSON or YAML transform.",
    )
    parser.add_argument("--source-pcd", required=True, help="Path to the source point cloud file.")
    parser.add_argument("--target-pcd", required=True, help="Path to the target point cloud file.")
    parser.add_argument("--transform", required=True, help="Path to a JSON or YAML transform from source to target.")
    parser.add_argument("--output-pcd", default="merged_output.pcd", help="Path to save the merged point cloud.")
    args = parser.parse_args()

    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)
    transform = load_initial_transform(args.transform)
    save_registered_point_cloud(source_cloud, target_cloud, transform, args.output_pcd)


if __name__ == "__main__":
    main()
