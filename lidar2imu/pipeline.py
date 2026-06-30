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

import numpy as np

from lidar2imu.evaluation.builder import build_metrics_output
from lidar2imu.models import CalibrationConfig, CalibrationDataset
from lidar2imu.solvers.dataset_split import split_holdout_dataset
from lidar2imu.solvers.orchestrator import run_calibration_once
from lidar2imu.solvers.robustness import (
    evaluate_full_prior_robustness,
    evaluate_holdout_repeatability,
    evaluate_planar_basin_stability,
)


def run_calibration(
    dataset: CalibrationDataset,
    config: CalibrationConfig | None = None,
    output_dir: str | None = None,
) -> dict:
    config = config or CalibrationConfig()
    calibration_dataset, holdout_dataset, holdout_plan = split_holdout_dataset(
        dataset, config
    )
    primary_result = run_calibration_once(
        calibration_dataset,
        config,
        np.asarray(calibration_dataset.initial_transform, dtype=float),
    )
    basin_stability = evaluate_planar_basin_stability(
        calibration_dataset,
        config,
        primary_result,
        run_solver_once=run_calibration_once,
    )
    full_prior_robustness = evaluate_full_prior_robustness(
        calibration_dataset,
        config,
        primary_result,
        run_solver_once=run_calibration_once,
    )
    holdout_repeatability = evaluate_holdout_repeatability(
        dataset,
        config,
        primary_result,
        run_solver_once=run_calibration_once,
        build_metrics_output=build_metrics_output,
    )
    metrics_output, evaluation_diagnostics = build_metrics_output(
        dataset=calibration_dataset,
        final_transform=primary_result["final_transform"],
        initial_transform=primary_result["initial_transform"],
        stages=primary_result["stages"],
        config=config,
        output_dir=output_dir or "",
        basin_stability=basin_stability,
        full_prior_robustness=full_prior_robustness,
        holdout_repeatability=holdout_repeatability,
        full_dataset=dataset,
        holdout_dataset=holdout_dataset,
        holdout_plan=holdout_plan,
    )

    return {
        "initial_transform": primary_result["initial_transform"],
        "final_transform": primary_result["final_transform"],
        "stages": primary_result["stages"],
        "metrics": metrics_output,
        "evaluation": evaluation_diagnostics,
    }
