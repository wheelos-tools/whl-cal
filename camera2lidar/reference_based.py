#!/usr/bin/env python

# Copyright 2025 WheelOS. All Rights Reserved.
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

"""DEPRECATED: `camera2lidar` merged into `lidar2camera`.

This compatibility wrapper is left in place for legacy invocations but users
should call the canonical CLI `lidar2camera-calibrate` or import
`lidar2camera.reference_pipeline` instead.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lidar2camera.reference_pipeline import (
    default_reference_config_payload,
    run_reference_calibration_from_config,
)


def main() -> None:
    config_path = Path("config.yaml")
    if not config_path.exists():
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(default_reference_config_payload(), file, sort_keys=False)
        print(
            f"Created default config at {config_path}. Review it, place synchronized image/pcd pairs under the configured data_directory, and rerun."
        )
        return
    run_reference_calibration_from_config(str(config_path))


if __name__ == "__main__":
    main()
