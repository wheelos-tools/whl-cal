from __future__ import annotations

# isort: off
from typing import Any

import numpy as np

from lidar2imu.models import CalibrationConfig, CalibrationDataset
from lidar2imu.solvers.motion_objective import motion_sample_window_ids
from lidar2imu.solvers.motion_objective import run_planar_motion_stage
from lidar2imu.solvers import motion_screening as _motion_screening
from lidar2imu.solvers import motion_subset_search as _motion_subset_search
from lidar2imu.solvers.motion_sufficiency import assess_motion_sufficiency
from lidar2imu.solvers.temporal_alignment import assess_temporal_alignment

# isort: on


def _combined_screening_summary(
    pre_screen_summary: dict[str, Any],
    consistency_summary: dict[str, Any],
    *,
    active_sample_count: int,
    sufficiency_summary: dict[str, Any] | None = None,
    temporal_alignment_summary: dict[str, Any] | None = None,
    reselection_summary: dict[str, Any] | None = None,
    local_search_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_sample_count = int(
        pre_screen_summary.get("input_sample_count", active_sample_count)
    )
    summary = {
        "input_sample_count": input_sample_count,
        "selected_sample_count": int(active_sample_count),
        "skipped_sample_count": int(max(input_sample_count - active_sample_count, 0)),
        "min_motion_rotation_deg": float(
            pre_screen_summary.get("min_motion_rotation_deg", 0.0)
        ),
        "min_registration_fitness": float(
            pre_screen_summary.get("min_registration_fitness", 0.0)
        ),
        "max_registration_inlier_rmse_m": float(
            pre_screen_summary.get("max_registration_inlier_rmse_m", 0.0)
        ),
        "skip_reason_counts": dict(pre_screen_summary.get("skip_reason_counts", {})),
        "pre_screen": pre_screen_summary,
        "consistency": consistency_summary,
    }
    if sufficiency_summary is not None:
        summary["sufficiency"] = sufficiency_summary
    if temporal_alignment_summary is not None:
        summary["temporal_alignment"] = temporal_alignment_summary
    if reselection_summary is not None:
        summary["reselection"] = reselection_summary
    if local_search_summary is not None:
        summary["local_search"] = local_search_summary
    return summary


def run_baseline_planar_stage(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    ground_stage: dict[str, Any],
    initial_transform: np.ndarray,
) -> dict[str, Any]:
    return run_planar_motion_stage(
        dataset.motion_samples,
        config,
        ground_stage,
        initial_transform,
        stage_family="baseline_planar",
        screening_summary=None,
    )


def run_gril_planar_stage(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    ground_stage: dict[str, Any],
    initial_transform: np.ndarray,
    *,
    stage_family_name: str = "gril_planar",
    seed_stage_family_name: str = "gril_planar_seed",
) -> dict[str, Any]:
    screened_motion_samples, pre_screen_summary = (
        _motion_screening.screen_motion_samples(dataset.motion_samples, config)
    )
    seed_planar = run_planar_motion_stage(
        screened_motion_samples,
        config,
        ground_stage,
        initial_transform,
        stage_family=seed_stage_family_name,
        screening_summary=pre_screen_summary,
    )
    consistent_motion_samples, consistency_summary = (
        _motion_screening.prune_inconsistent_motion_samples(
            screened_motion_samples,
            rotation=np.asarray(seed_planar["rotation"], dtype=float),
            translation=np.asarray(seed_planar["translation"], dtype=float),
            config=config,
        )
    )
    reselected_motion_samples, reselection_summary = (
        _motion_screening.reselect_motion_samples_from_pool(
            consistent_motion_samples,
            dataset.motion_candidate_pool,
            rotation=np.asarray(seed_planar["rotation"], dtype=float),
            translation=np.asarray(seed_planar["translation"], dtype=float),
            config=config,
            target_sample_count=len(screened_motion_samples),
        )
    )

    candidate_pool_for_sufficiency = (
        list(dataset.motion_candidate_pool)
        if dataset.motion_candidate_pool
        else list(screened_motion_samples)
    )
    pre_local_search_sufficiency = assess_motion_sufficiency(
        reselected_motion_samples,
        candidate_pool_for_sufficiency,
        config,
    )

    if pre_local_search_sufficiency["ready_for_local_search"]:
        optimized_motion_samples, local_search_summary = (
            _motion_subset_search.improve_motion_samples_with_local_search(
                dataset,
                config,
                ground_stage,
                initial_transform,
                reselected_motion_samples,
                locked_window_ids=(
                    set(pre_local_search_sufficiency["locked_window_ids"])
                    or _motion_subset_search.anchor_window_ids(
                        consistent_motion_samples
                    )
                ),
            )
        )
    else:
        optimized_motion_samples = list(reselected_motion_samples)
        local_search_summary = {
            "enabled": False,
            "reason": "motion_sufficiency_gate",
            "gating_reasons": list(
                pre_local_search_sufficiency.get("local_search_reasons", [])
            ),
            "locked_window_ids": list(
                pre_local_search_sufficiency.get("locked_window_ids", [])
            ),
            "final_window_ids": motion_sample_window_ids(optimized_motion_samples),
        }
    local_search_summary["sufficiency"] = pre_local_search_sufficiency

    final_sufficiency = assess_motion_sufficiency(
        optimized_motion_samples,
        candidate_pool_for_sufficiency,
        config,
    )
    final_temporal_alignment = assess_temporal_alignment(
        optimized_motion_samples,
        candidate_pool_for_sufficiency,
    )
    screening_summary = _combined_screening_summary(
        pre_screen_summary,
        consistency_summary,
        active_sample_count=len(optimized_motion_samples),
        sufficiency_summary=final_sufficiency,
        temporal_alignment_summary=final_temporal_alignment,
        reselection_summary=reselection_summary,
        local_search_summary=local_search_summary,
    )
    final_planar = run_planar_motion_stage(
        optimized_motion_samples,
        config,
        ground_stage,
        initial_transform,
        stage_family=stage_family_name,
        screening_summary=screening_summary,
    )
    final_planar["summary"]["consistency_filter"] = consistency_summary
    final_planar["summary"]["reselection"] = reselection_summary
    final_planar["summary"]["local_search"] = local_search_summary
    final_planar["summary"]["sufficiency"] = final_sufficiency
    final_planar["summary"]["temporal_alignment"] = final_temporal_alignment
    return final_planar
