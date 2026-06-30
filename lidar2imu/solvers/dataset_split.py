from __future__ import annotations

import numpy as np

from lidar2imu.models import CalibrationConfig, CalibrationDataset, MotionSample


def clone_dataset_with_motion_samples(
    dataset: CalibrationDataset,
    motion_samples: list[MotionSample],
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
        motion_candidate_pool=list(dataset.motion_candidate_pool),
        metadata=metadata,
    )


def split_holdout_dataset(
    dataset: CalibrationDataset, config: CalibrationConfig
) -> tuple[CalibrationDataset, CalibrationDataset | None, dict]:
    holdout_every_n = max(int(config.metrics_holdout_every_n), 0)
    default_offset = max(holdout_every_n - 1, 0)
    return split_holdout_dataset_with_offset(
        dataset, config, holdout_offset=default_offset
    )


def split_holdout_dataset_with_offset(
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
    calibration_dataset = clone_dataset_with_motion_samples(
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
    holdout_dataset = clone_dataset_with_motion_samples(
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
