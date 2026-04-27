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
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import (
    refine_joint_solution,
    transform_delta_metrics,
    solve_ground_translation,
    solve_roll_pitch_from_ground,
    solve_translation_from_motion,
    solve_yaw_from_motion,
    transform_from_components,
    yaw_roll_pitch_from_matrix,
)
from lidar2imu.metrics import build_metrics_output
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def _count_turns(dataset: CalibrationDataset) -> tuple[int, int]:
    signed_yaw_deg = []
    for sample in dataset.motion_samples:
        try:
            signed_yaw_deg.append(
                float(
                    np.degrees(
                        R.from_matrix(sample.imu_delta_rotation).as_euler(
                            "ZYX", degrees=False
                        )[0]
                    )
                )
            )
        except ValueError:
            signed_yaw_deg.append(0.0)
    left_turn_count = int(sum(1 for value in signed_yaw_deg if value > 0.5))
    right_turn_count = int(sum(1 for value in signed_yaw_deg if value < -0.5))
    return left_turn_count, right_turn_count


def _resolve_planar_motion_policy(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    motion_rotation_stage: dict,
) -> dict:
    left_turn_count, right_turn_count = _count_turns(dataset)
    yaw_observability = motion_rotation_stage.get("observability", {})
    yaw_observability_reasons = list(yaw_observability.get("reasons", []))
    weak_planar_reasons = []
    if (
        min(left_turn_count, right_turn_count)
        < config.metrics_min_turn_count_per_direction
    ):
        weak_planar_reasons.append("turn_imbalance")
    if yaw_observability.get("degenerate", False):
        weak_planar_reasons.append("yaw_rotation_degenerate")

    requested_policy = config.planar_motion_policy
    applied_policy = requested_policy
    if requested_policy == "auto":
        applied_policy = "freeze_xyyaw" if weak_planar_reasons else "free"

    return {
        "requested": requested_policy,
        "applied": applied_policy,
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "weak_planar_reasons": weak_planar_reasons,
        "yaw_observability_reasons": yaw_observability_reasons,
        "locked_components": (
            ["yaw", "x", "y"] if applied_policy == "freeze_xyyaw" else []
        ),
    }


def _clone_dataset_with_motion_samples(
    dataset: CalibrationDataset,
    motion_samples: list,
    *,
    metadata_updates: dict | None = None,
) -> CalibrationDataset:
    metadata = dict(dataset.metadata)
    if metadata_updates:
        metadata.update(metadata_updates)
    return CalibrationDataset(
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        ground_samples=list(dataset.ground_samples),
        motion_samples=list(motion_samples),
        initial_transform=np.asarray(dataset.initial_transform, dtype=float),
        extraction_transform=(
            None
            if dataset.extraction_transform is None
            else np.asarray(dataset.extraction_transform, dtype=float)
        ),
        reference_transform=(
            None
            if dataset.reference_transform is None
            else np.asarray(dataset.reference_transform, dtype=float)
        ),
        metadata=metadata,
    )


def _split_holdout_dataset(
    dataset: CalibrationDataset, config: CalibrationConfig
) -> tuple[CalibrationDataset, CalibrationDataset | None, dict]:
    holdout_every_n = max(int(config.metrics_holdout_every_n), 0)
    default_offset = max(holdout_every_n - 1, 0)
    return _split_holdout_dataset_with_offset(
        dataset, config, holdout_offset=default_offset
    )


def _split_holdout_dataset_with_offset(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    *,
    holdout_offset: int,
) -> tuple[CalibrationDataset, CalibrationDataset | None, dict]:
    ordered_motion_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    total_motion_samples = len(ordered_motion_samples)
    holdout_every_n = max(int(config.metrics_holdout_every_n), 0)
    normalized_offset = 0
    if holdout_every_n >= 1:
        normalized_offset = int(holdout_offset) % holdout_every_n
    holdout_plan = {
        "enabled": False,
        "strategy": "every_nth_motion_sample",
        "every_n": holdout_every_n,
        "offset": normalized_offset,
        "total_motion_samples": total_motion_samples,
        "calibration_motion_samples": total_motion_samples,
        "holdout_motion_samples": 0,
        "reason": "holdout_disabled",
    }
    if holdout_every_n < 2:
        holdout_plan["reason"] = "invalid_holdout_every_n"
        return dataset, None, holdout_plan

    if total_motion_samples < (
        config.min_motion_samples + config.metrics_holdout_min_motion_samples
    ):
        holdout_plan["reason"] = "insufficient_motion_samples"
        return dataset, None, holdout_plan

    holdout_indices = [
        index
        for index in range(total_motion_samples)
        if index % holdout_every_n == normalized_offset
    ]
    while (
        total_motion_samples - len(holdout_indices) < config.min_motion_samples
        and holdout_indices
    ):
        holdout_indices.pop()
    if len(holdout_indices) < config.metrics_holdout_min_motion_samples:
        holdout_plan["reason"] = "insufficient_holdout_samples"
        return dataset, None, holdout_plan

    holdout_index_set = set(holdout_indices)
    calibration_motion_samples = [
        sample
        for index, sample in enumerate(ordered_motion_samples)
        if index not in holdout_index_set
    ]
    holdout_motion_samples = [
        sample
        for index, sample in enumerate(ordered_motion_samples)
        if index in holdout_index_set
    ]
    holdout_plan = {
        "enabled": True,
        "strategy": "every_nth_motion_sample",
        "every_n": holdout_every_n,
        "offset": normalized_offset,
        "total_motion_samples": total_motion_samples,
        "calibration_motion_samples": len(calibration_motion_samples),
        "holdout_motion_samples": len(holdout_motion_samples),
        "holdout_indices": [int(index) for index in holdout_indices],
        "reason": None,
    }
    calibration_dataset = _clone_dataset_with_motion_samples(
        dataset,
        calibration_motion_samples,
        metadata_updates={
            "motion_holdout_split": "calibration",
            "motion_holdout_every_n": holdout_every_n,
            "motion_holdout_offset": normalized_offset,
            "motion_holdout_total_samples": total_motion_samples,
            "motion_holdout_count": len(holdout_motion_samples),
        },
    )
    holdout_dataset = _clone_dataset_with_motion_samples(
        dataset,
        holdout_motion_samples,
        metadata_updates={
            "motion_holdout_split": "holdout",
            "motion_holdout_every_n": holdout_every_n,
            "motion_holdout_offset": normalized_offset,
            "motion_holdout_total_samples": total_motion_samples,
            "motion_holdout_count": len(holdout_motion_samples),
        },
    )
    return calibration_dataset, holdout_dataset, holdout_plan


def _summarize_numeric(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    series = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "span": float(np.max(series) - np.min(series)),
        "p95": float(np.percentile(series, 95)),
    }


def _evaluate_holdout_repeatability(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    primary_result: dict,
) -> dict | None:
    holdout_every_n = max(int(config.metrics_holdout_every_n), 0)
    repeat_count = max(int(config.metrics_holdout_repeat_count), 0)
    min_repeat_evaluations = max(int(config.metrics_holdout_min_repeat_evaluations), 1)
    if holdout_every_n < 2 or repeat_count < min_repeat_evaluations:
        return None

    evaluated_offsets = min(holdout_every_n, repeat_count)
    trials = []
    trial_final_transforms: list[np.ndarray] = []
    for holdout_offset in range(evaluated_offsets):
        calibration_dataset, holdout_dataset, holdout_plan = (
            _split_holdout_dataset_with_offset(
                dataset,
                config,
                holdout_offset=holdout_offset,
            )
        )
        if not holdout_plan.get("enabled") or holdout_dataset is None:
            continue

        trial_result = _run_calibration_once(
            calibration_dataset,
            config,
            np.asarray(calibration_dataset.initial_transform, dtype=float),
        )
        trial_metrics, _ = build_metrics_output(
            dataset=calibration_dataset,
            final_transform=trial_result["final_transform"],
            initial_transform=trial_result["initial_transform"],
            stages=trial_result["stages"],
            config=config,
            output_dir="",
            full_dataset=dataset,
            holdout_dataset=holdout_dataset,
            holdout_plan=holdout_plan,
        )
        holdout_validation = (
            trial_metrics.get("fine_metrics", {}).get("holdout_validation", {}) or {}
        )
        final_transform = np.asarray(trial_result["final_transform"], dtype=float)
        yaw_rad, roll_rad, pitch_rad = yaw_roll_pitch_from_matrix(final_transform)
        trial_final_transforms.append(final_transform)
        trials.append(
            {
                "offset": int(holdout_plan.get("offset", holdout_offset)),
                "status": holdout_validation.get("status", "unknown"),
                "primary_cause": holdout_validation.get("primary_cause"),
                "holdout_motion_samples": holdout_validation.get(
                    "holdout_motion_samples"
                ),
                "calibration_motion_samples": holdout_validation.get(
                    "calibration_motion_samples"
                ),
                "ratios": holdout_validation.get("ratios"),
                "checks": holdout_validation.get("checks"),
                "final_delta_to_primary": transform_delta_metrics(
                    primary_result["final_transform"], final_transform
                ),
                "final_translation_m": {
                    "x": float(final_transform[0, 3]),
                    "y": float(final_transform[1, 3]),
                    "z": float(final_transform[2, 3]),
                },
                "final_euler_deg": {
                    "yaw": float(np.degrees(yaw_rad)),
                    "roll": float(np.degrees(roll_rad)),
                    "pitch": float(np.degrees(pitch_rad)),
                },
            }
        )

    if len(trials) < min_repeat_evaluations:
        return None

    translation_threshold = float(config.metrics_repeatability_warning_translation_m)
    rotation_threshold = float(config.metrics_repeatability_warning_rotation_deg)
    distinct_solution_count = _cluster_final_transforms(
        trial_final_transforms,
        translation_threshold_m=translation_threshold,
        rotation_threshold_deg=rotation_threshold,
    )
    max_pairwise_translation = 0.0
    max_pairwise_rotation = 0.0
    for left_index, left_transform in enumerate(trial_final_transforms):
        for right_transform in trial_final_transforms[left_index + 1 :]:
            delta = transform_delta_metrics(left_transform, right_transform)
            max_pairwise_translation = max(
                max_pairwise_translation, float(delta["translation_norm_m"])
            )
            max_pairwise_rotation = max(
                max_pairwise_rotation, float(delta["rotation_deg"])
            )

    pass_count = int(sum(1 for trial in trials if trial["status"] == "pass"))
    warning_count = int(sum(1 for trial in trials if trial["status"] == "warning"))
    unknown_count = int(sum(1 for trial in trials if trial["status"] == "unknown"))
    unstable_offsets = [
        int(trial["offset"])
        for trial in trials
        if float(trial["final_delta_to_primary"]["translation_norm_m"])
        > translation_threshold
        or float(trial["final_delta_to_primary"]["rotation_deg"]) > rotation_threshold
    ]

    if warning_count == 0 and distinct_solution_count <= 1:
        status = "pass"
        primary_cause = "stable_repeated_holdout"
        recommendations: list[str] = []
    elif distinct_solution_count > 1:
        status = "warning"
        primary_cause = "repeat_split_solution_instability"
        recommendations = [
            "Different holdout offsets converge to materially different final solutions; do not treat one split as release-quality evidence.",
            "Require repeatability across bag families or tighten map-side constraints before promoting this configuration.",
        ]
    else:
        status = "warning"
        primary_cause = "repeat_split_generalization_gap"
        recommendations = [
            "At least one holdout offset degrades materially relative to the calibration subset; require repeated-split stability before trusting free x/y/yaw.",
            "Prefer map settings that keep holdout ratios stable across offsets instead of optimizing one deterministic split only.",
        ]

    yaw_values_rad = np.unwrap(
        np.asarray(
            [np.radians(float(trial["final_euler_deg"]["yaw"])) for trial in trials],
            dtype=float,
        )
    )
    roll_values_deg = [float(trial["final_euler_deg"]["roll"]) for trial in trials]
    pitch_values_deg = [float(trial["final_euler_deg"]["pitch"]) for trial in trials]
    x_values_m = [float(trial["final_translation_m"]["x"]) for trial in trials]
    y_values_m = [float(trial["final_translation_m"]["y"]) for trial in trials]
    z_values_m = [float(trial["final_translation_m"]["z"]) for trial in trials]
    translation_delta_norms = [
        float(trial["final_delta_to_primary"]["translation_norm_m"]) for trial in trials
    ]
    rotation_delta_norms = [
        float(trial["final_delta_to_primary"]["rotation_deg"]) for trial in trials
    ]

    return {
        "status": status,
        "primary_cause": primary_cause,
        "trial_count": len(trials),
        "evaluated_offsets": [int(trial["offset"]) for trial in trials],
        "pass_count": pass_count,
        "warning_count": warning_count,
        "unknown_count": unknown_count,
        "pass_fraction": float(pass_count / len(trials)),
        "distinct_solution_count": distinct_solution_count,
        "unstable_offsets": unstable_offsets,
        "translation_threshold_m": translation_threshold,
        "rotation_threshold_deg": rotation_threshold,
        "max_pairwise_delta": {
            "translation_norm_m": float(max_pairwise_translation),
            "rotation_deg": float(max_pairwise_rotation),
        },
        "delta_to_primary_summary": {
            "translation_norm_m": _summarize_numeric(translation_delta_norms),
            "rotation_deg": _summarize_numeric(rotation_delta_norms),
        },
        "uncertainty_summary": {
            "source": "repeated_holdout_offsets",
            "trial_count": len(trials),
            "final_translation_m": {
                "x": _summarize_numeric(x_values_m),
                "y": _summarize_numeric(y_values_m),
                "z": _summarize_numeric(z_values_m),
            },
            "final_euler_deg": {
                "yaw": _summarize_numeric(list(np.degrees(yaw_values_rad))),
                "roll": _summarize_numeric(roll_values_deg),
                "pitch": _summarize_numeric(pitch_values_deg),
            },
            "delta_to_primary": {
                "translation_norm_m": _summarize_numeric(translation_delta_norms),
                "rotation_deg": _summarize_numeric(rotation_delta_norms),
            },
        },
        "recommendations": recommendations,
        "trials": trials,
    }


def _run_calibration_once(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    initial_transform: np.ndarray,
) -> dict:
    initial_transform = np.asarray(initial_transform, dtype=float)
    initial_yaw, initial_roll, initial_pitch = yaw_roll_pitch_from_matrix(
        initial_transform
    )
    initial_translation = np.asarray(initial_transform[:3, 3], dtype=float)

    ground_rotation, ground_orientation_stage = solve_roll_pitch_from_ground(
        dataset.ground_samples,
        initial_roll=initial_roll,
        initial_pitch=initial_pitch,
        config=config,
    )
    ground_translation, ground_translation_stage = solve_ground_translation(
        dataset.ground_samples,
        rotation=ground_rotation,
        initial_translation=initial_translation,
    )
    yaw, motion_rotation_stage = solve_yaw_from_motion(
        dataset.motion_samples,
        fixed_roll=ground_orientation_stage["parameters"]["roll_rad"],
        fixed_pitch=ground_orientation_stage["parameters"]["pitch_rad"],
        initial_yaw=initial_yaw,
        config=config,
    )
    motion_rotation = transform_from_components(
        yaw=yaw,
        roll=ground_orientation_stage["parameters"]["roll_rad"],
        pitch=ground_orientation_stage["parameters"]["pitch_rad"],
        translation=np.zeros(3, dtype=float),
    )[:3, :3]
    solver_policy = _resolve_planar_motion_policy(
        dataset=dataset,
        config=config,
        motion_rotation_stage=motion_rotation_stage,
    )
    translation_locked_axes = (
        (True, True, False)
        if solver_policy["applied"] == "freeze_xyyaw"
        else (False, False, False)
    )
    motion_translation_initial = np.asarray(ground_translation, dtype=float).reshape(3)
    if solver_policy["applied"] == "freeze_xyyaw":
        motion_translation_initial[:2] = initial_translation[:2]
    motion_translation, motion_translation_stage = solve_translation_from_motion(
        dataset.motion_samples,
        rotation=motion_rotation,
        initial_translation=motion_translation_initial,
        locked_axes=translation_locked_axes,
    )
    joint_yaw = initial_yaw if solver_policy["applied"] == "freeze_xyyaw" else yaw

    joint_initial = np.array(
        [
            joint_yaw,
            ground_orientation_stage["parameters"]["roll_rad"],
            ground_orientation_stage["parameters"]["pitch_rad"],
            motion_translation[0],
            motion_translation[1],
            motion_translation[2],
        ],
        dtype=float,
    )
    final_transform, joint_stage = refine_joint_solution(
        dataset.ground_samples,
        dataset.motion_samples,
        initial_params=joint_initial,
        config=config,
        locked_mask=(
            solver_policy["applied"] == "freeze_xyyaw",
            False,
            False,
            solver_policy["applied"] == "freeze_xyyaw",
            solver_policy["applied"] == "freeze_xyyaw",
            False,
        ),
    )
    stages = {
        "ground_orientation": ground_orientation_stage,
        "ground_translation": ground_translation_stage,
        "motion_rotation": motion_rotation_stage,
        "motion_translation": motion_translation_stage,
        "solver_policy": solver_policy,
        "joint": joint_stage,
    }
    return {
        "initial_transform": initial_transform,
        "final_transform": final_transform,
        "stages": stages,
    }


def _build_multistart_seed(
    transform: np.ndarray,
    *,
    yaw_delta_deg: float = 0.0,
    roll_delta_deg: float = 0.0,
    pitch_delta_deg: float = 0.0,
    x_delta_m: float = 0.0,
    y_delta_m: float = 0.0,
    z_delta_m: float = 0.0,
) -> np.ndarray:
    yaw, roll, pitch = yaw_roll_pitch_from_matrix(transform)
    translation = np.asarray(transform[:3, 3], dtype=float).copy()
    translation[0] += float(x_delta_m)
    translation[1] += float(y_delta_m)
    translation[2] += float(z_delta_m)
    return transform_from_components(
        yaw=yaw + np.radians(float(yaw_delta_deg)),
        roll=roll + np.radians(float(roll_delta_deg)),
        pitch=pitch + np.radians(float(pitch_delta_deg)),
        translation=translation,
    )


def _append_unique_seed(
    seeds: list[dict], name: str, candidate: np.ndarray, *, group: str
) -> None:
    for existing in seeds:
        delta = transform_delta_metrics(existing["transform"], candidate)
        if delta["translation_norm_m"] <= 1e-9 and delta["rotation_deg"] <= 1e-9:
            return
    seeds.append(
        {
            "name": name,
            "group": group,
            "transform": np.asarray(candidate, dtype=float),
        }
    )


def _cluster_final_transforms(
    transforms: list[np.ndarray],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> int:
    representatives: list[np.ndarray] = []
    for transform in transforms:
        matched = False
        for representative in representatives:
            delta = transform_delta_metrics(representative, transform)
            if (
                delta["translation_norm_m"] <= translation_threshold_m
                and delta["rotation_deg"] <= rotation_threshold_deg
            ):
                matched = True
                break
        if not matched:
            representatives.append(transform)
    return len(representatives)


def _evaluate_planar_basin_stability(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    primary_result: dict,
) -> dict | None:
    reference_transform = dataset.reference_transform
    if reference_transform is None:
        return None

    perturb_translation = config.metrics_multistart_translation_perturbation_m
    perturb_yaw = config.metrics_multistart_yaw_perturbation_deg
    seeds: list[dict] = []
    _append_unique_seed(
        seeds,
        "input_initial",
        np.asarray(dataset.initial_transform, dtype=float),
        group="input",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference",
        np.asarray(reference_transform, dtype=float),
        group="reference",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_perturbed_plus",
        _build_multistart_seed(
            reference_transform,
            yaw_delta_deg=perturb_yaw,
            x_delta_m=perturb_translation,
            y_delta_m=-perturb_translation,
        ),
        group="planar",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_perturbed_minus",
        _build_multistart_seed(
            reference_transform,
            yaw_delta_deg=-perturb_yaw,
            x_delta_m=-perturb_translation,
            y_delta_m=perturb_translation,
        ),
        group="planar",
    )

    trials = []
    for seed in seeds:
        name = str(seed["name"])
        seed_transform = np.asarray(seed["transform"], dtype=float)
        trial_result = _run_calibration_once(dataset, config, seed_transform)
        final_transform = trial_result["final_transform"]
        motion_observability = (
            trial_result["stages"].get("motion_rotation", {}).get("observability", {})
        )
        trials.append(
            {
                "name": name,
                "group": seed["group"],
                "initial_delta_to_reference": transform_delta_metrics(
                    reference_transform, seed_transform
                ),
                "final_delta_to_reference": transform_delta_metrics(
                    reference_transform, final_transform
                ),
                "final_delta_to_primary": transform_delta_metrics(
                    primary_result["final_transform"], final_transform
                ),
                "solver_policy": trial_result["stages"]
                .get("solver_policy", {})
                .get("applied"),
                "max_cost_ratio": (motion_observability.get("cost_scan") or {}).get(
                    "max_cost_ratio"
                ),
                "within_5pct_span_deg": (
                    motion_observability.get("cost_scan") or {}
                ).get("within_5pct_span_deg"),
                "final_transform": final_transform,
            }
        )

    translation_threshold = config.metrics_reference_warning_translation_m
    rotation_threshold = config.metrics_reference_warning_rotation_deg
    final_transforms = [trial["final_transform"] for trial in trials]
    distinct_solution_count = _cluster_final_transforms(
        final_transforms,
        translation_threshold_m=translation_threshold,
        rotation_threshold_deg=rotation_threshold,
    )
    max_pairwise_translation = 0.0
    max_pairwise_rotation = 0.0
    for left_index, left_transform in enumerate(final_transforms):
        for right_transform in final_transforms[left_index + 1 :]:
            delta = transform_delta_metrics(left_transform, right_transform)
            max_pairwise_translation = max(
                max_pairwise_translation, delta["translation_norm_m"]
            )
            max_pairwise_rotation = max(max_pairwise_rotation, delta["rotation_deg"])

    reference_consistent_trial_count = int(
        sum(
            1
            for trial in trials
            if trial["final_delta_to_reference"]["translation_norm_m"]
            <= translation_threshold
            and trial["final_delta_to_reference"]["rotation_deg"] <= rotation_threshold
        )
    )
    status = "pass" if distinct_solution_count <= 1 else "warning"
    if status == "pass":
        primary_cause = "single_planar_basin"
        recommendations: list[str] = []
    elif 0 < reference_consistent_trial_count < len(trials):
        primary_cause = "multiple_planar_basins"
        recommendations = [
            "Nearby planar initial priors converge to different final basins; do not accept a single free-planar result as stable yet.",
            "Prefer the trusted-reference-consistent basin and require cross-bag or holdout evidence before overriding it.",
            "If this persists, strengthen map-side constraints or add reference-aware basin selection instead of relying on one local optimum.",
        ]
    else:
        primary_cause = "reference_inconsistent_basin_family"
        recommendations = [
            "All checked nearby starts stay away from the trusted reference basin; treat the current map objective as systematically biased on this bag.",
            "Do not accept free planar release from this configuration without an external measurement or stronger map-side validation.",
        ]

    for trial in trials:
        trial.pop("final_transform", None)
    return {
        "status": status,
        "primary_cause": primary_cause,
        "trial_count": len(trials),
        "distinct_solution_count": distinct_solution_count,
        "reference_consistent_trial_count": reference_consistent_trial_count,
        "translation_threshold_m": float(translation_threshold),
        "rotation_threshold_deg": float(rotation_threshold),
        "max_pairwise_delta": {
            "translation_norm_m": float(max_pairwise_translation),
            "rotation_deg": float(max_pairwise_rotation),
        },
        "recommendations": recommendations,
        "trials": trials,
    }


def _evaluate_full_prior_robustness(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    primary_result: dict,
) -> dict | None:
    reference_transform = dataset.reference_transform
    if reference_transform is None:
        return None

    perturb_translation = config.metrics_multistart_translation_perturbation_m
    perturb_yaw = config.metrics_multistart_yaw_perturbation_deg
    perturb_vertical = config.metrics_multistart_vertical_perturbation_m
    perturb_rp = config.metrics_multistart_roll_pitch_perturbation_deg

    seeds: list[dict] = []
    _append_unique_seed(
        seeds,
        "input_initial",
        np.asarray(dataset.initial_transform, dtype=float),
        group="input",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference",
        np.asarray(reference_transform, dtype=float),
        group="reference",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_planar_plus",
        _build_multistart_seed(
            reference_transform,
            yaw_delta_deg=perturb_yaw,
            x_delta_m=perturb_translation,
            y_delta_m=-perturb_translation,
        ),
        group="planar",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_planar_minus",
        _build_multistart_seed(
            reference_transform,
            yaw_delta_deg=-perturb_yaw,
            x_delta_m=-perturb_translation,
            y_delta_m=perturb_translation,
        ),
        group="planar",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_vertical_plus",
        _build_multistart_seed(reference_transform, z_delta_m=perturb_vertical),
        group="vertical",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_vertical_minus",
        _build_multistart_seed(reference_transform, z_delta_m=-perturb_vertical),
        group="vertical",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_roll_plus",
        _build_multistart_seed(reference_transform, roll_delta_deg=perturb_rp),
        group="attitude",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_roll_minus",
        _build_multistart_seed(reference_transform, roll_delta_deg=-perturb_rp),
        group="attitude",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_pitch_plus",
        _build_multistart_seed(reference_transform, pitch_delta_deg=perturb_rp),
        group="attitude",
    )
    _append_unique_seed(
        seeds,
        "trusted_reference_pitch_minus",
        _build_multistart_seed(reference_transform, pitch_delta_deg=-perturb_rp),
        group="attitude",
    )

    trials = []
    for seed in seeds:
        name = str(seed["name"])
        seed_transform = np.asarray(seed["transform"], dtype=float)
        trial_result = _run_calibration_once(dataset, config, seed_transform)
        final_transform = trial_result["final_transform"]
        trials.append(
            {
                "name": name,
                "group": seed["group"],
                "initial_delta_to_reference": transform_delta_metrics(
                    reference_transform, seed_transform
                ),
                "final_delta_to_reference": transform_delta_metrics(
                    reference_transform, final_transform
                ),
                "final_delta_to_primary": transform_delta_metrics(
                    primary_result["final_transform"], final_transform
                ),
                "solver_policy": trial_result["stages"]
                .get("solver_policy", {})
                .get("applied"),
                "final_transform": final_transform,
            }
        )

    translation_threshold = config.metrics_reference_warning_translation_m
    rotation_threshold = config.metrics_reference_warning_rotation_deg
    final_transforms = [trial["final_transform"] for trial in trials]
    distinct_solution_count = _cluster_final_transforms(
        final_transforms,
        translation_threshold_m=translation_threshold,
        rotation_threshold_deg=rotation_threshold,
    )
    max_pairwise_translation = 0.0
    max_pairwise_rotation = 0.0
    for left_index, left_transform in enumerate(final_transforms):
        for right_transform in final_transforms[left_index + 1 :]:
            delta = transform_delta_metrics(left_transform, right_transform)
            max_pairwise_translation = max(
                max_pairwise_translation, delta["translation_norm_m"]
            )
            max_pairwise_rotation = max(max_pairwise_rotation, delta["rotation_deg"])

    primary_consistent_trial_count = int(
        sum(
            1
            for trial in trials
            if trial["final_delta_to_primary"]["translation_norm_m"]
            <= translation_threshold
            and trial["final_delta_to_primary"]["rotation_deg"] <= rotation_threshold
        )
    )
    unstable_groups = sorted(
        {
            str(trial["group"])
            for trial in trials
            if trial["final_delta_to_primary"]["translation_norm_m"]
            > translation_threshold
            or trial["final_delta_to_primary"]["rotation_deg"] > rotation_threshold
        }
    )
    group_summary = {}
    for group in sorted({str(seed["group"]) for seed in seeds}):
        group_trials = [trial for trial in trials if trial["group"] == group]
        group_summary[group] = {
            "trial_count": len(group_trials),
            "primary_consistent_trial_count": int(
                sum(
                    1
                    for trial in group_trials
                    if trial["final_delta_to_primary"]["translation_norm_m"]
                    <= translation_threshold
                    and trial["final_delta_to_primary"]["rotation_deg"]
                    <= rotation_threshold
                )
            ),
        }

    status = "pass" if distinct_solution_count <= 1 else "warning"
    if status == "pass":
        primary_cause = "single_full_prior_basin"
        recommendations: list[str] = []
    elif any(group in {"vertical", "attitude"} for group in unstable_groups):
        primary_cause = "vertical_or_attitude_prior_sensitivity"
        recommendations = [
            "Perturbing z/roll/pitch changes the final solution family; do not treat this configuration as fully industrialized 6DoF prior-robust yet.",
            "Before trusting the released extrinsics, compare candidate extraction geometries and verify sensor height / installation attitude against external measurements.",
            "Strengthen full-6DoF acceptance with repeated replays before promoting this bag as a production reference.",
        ]
    elif "planar" in unstable_groups:
        primary_cause = "planar_prior_sensitivity"
        recommendations = [
            "Planar perturbations still lead to multiple final solution families; do not treat one free-planar result as a release-quality answer yet.",
            "Prefer the trusted-reference-consistent basin and require repeatability across bags or map settings before overriding it.",
        ]
    else:
        primary_cause = "mixed_prior_sensitivity"
        recommendations = [
            "Multiple prior perturbations converge to different final families; treat this run as sensitivity-limited rather than production-robust.",
            "Compare candidate priors and extraction settings before trusting a single calibration output.",
        ]

    for trial in trials:
        trial.pop("final_transform", None)
    return {
        "status": status,
        "primary_cause": primary_cause,
        "trial_count": len(trials),
        "distinct_solution_count": distinct_solution_count,
        "primary_consistent_trial_count": primary_consistent_trial_count,
        "unstable_groups": unstable_groups,
        "group_summary": group_summary,
        "translation_threshold_m": float(translation_threshold),
        "rotation_threshold_deg": float(rotation_threshold),
        "max_pairwise_delta": {
            "translation_norm_m": float(max_pairwise_translation),
            "rotation_deg": float(max_pairwise_rotation),
        },
        "recommendations": recommendations,
        "trials": trials,
    }


def run_calibration(
    dataset: CalibrationDataset,
    config: CalibrationConfig | None = None,
    output_dir: str | None = None,
) -> dict:
    config = config or CalibrationConfig()
    calibration_dataset, holdout_dataset, holdout_plan = _split_holdout_dataset(
        dataset, config
    )
    primary_result = _run_calibration_once(
        calibration_dataset,
        config,
        np.asarray(calibration_dataset.initial_transform, dtype=float),
    )
    basin_stability = _evaluate_planar_basin_stability(
        calibration_dataset, config, primary_result
    )
    full_prior_robustness = _evaluate_full_prior_robustness(
        calibration_dataset, config, primary_result
    )
    holdout_repeatability = _evaluate_holdout_repeatability(
        dataset, config, primary_result
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
