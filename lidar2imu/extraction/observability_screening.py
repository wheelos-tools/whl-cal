from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.extraction.motion_windows import heading_bin, motion_turn_sign


def registered_candidate_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(candidate["start_meta"].timestamp_ns),
        int(candidate["end_meta"].timestamp_ns),
        int(candidate["stride"]),
    )


def _registered_candidate_key_payload(candidate: dict[str, Any]) -> dict[str, int]:
    start_timestamp_ns, end_timestamp_ns, frame_stride = registered_candidate_key(
        candidate
    )
    return {
        "start_timestamp_ns": int(start_timestamp_ns),
        "end_timestamp_ns": int(end_timestamp_ns),
        "frame_stride": int(frame_stride),
    }


def _candidate_key_from_payload(payload: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(payload["start_timestamp_ns"]),
        int(payload["end_timestamp_ns"]),
        int(payload["frame_stride"]),
    )


def _sort_candidate_key_payloads(
    key_payloads: list[dict[str, Any]],
) -> list[dict[str, int]]:
    return sorted(
        [
            {
                "start_timestamp_ns": int(payload["start_timestamp_ns"]),
                "end_timestamp_ns": int(payload["end_timestamp_ns"]),
                "frame_stride": int(payload["frame_stride"]),
            }
            for payload in key_payloads
        ],
        key=lambda payload: _candidate_key_from_payload(payload),
    )


def _segment_candidate_key_tuples(
    segment: dict[str, Any],
) -> set[tuple[int, int, int]]:
    key_tuples: set[tuple[int, int, int]] = set()
    for payload in cast(list[Any], segment.get("candidate_keys", [])):
        if not isinstance(payload, dict):
            continue
        try:
            key_tuples.add(_candidate_key_from_payload(cast(dict[str, Any], payload)))
        except (KeyError, TypeError, ValueError):
            continue
    return key_tuples


def _candidate_screening_decisions(
    *,
    candidates_by_key: dict[tuple[int, int, int], dict[str, Any]],
    segment_summaries: list[dict[str, Any]],
    selected_segments: list[dict[str, Any]],
    selected_key_tuples: set[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    segment_by_id: dict[int, dict[str, Any]] = {}
    candidate_segment_ids: dict[tuple[int, int, int], set[int]] = {}
    for segment in segment_summaries:
        try:
            segment_id = int(segment["segment_id"])
        except (KeyError, TypeError, ValueError):
            continue
        segment_by_id[segment_id] = segment
        for key_tuple in _segment_candidate_key_tuples(segment):
            candidate_segment_ids.setdefault(key_tuple, set()).add(segment_id)

    selected_segment_ids = {
        int(segment["segment_id"])
        for segment in selected_segments
        if isinstance(segment, dict) and "segment_id" in segment
    }
    decisions: list[dict[str, Any]] = []
    for key_tuple in sorted(candidates_by_key):
        candidate = candidates_by_key[key_tuple]
        segment_ids = sorted(candidate_segment_ids.get(key_tuple, set()))
        segment_rows = [
            segment_by_id[segment_id]
            for segment_id in segment_ids
            if segment_id in segment_by_id
        ]
        passing_segment_ids = [
            int(segment["segment_id"])
            for segment in segment_rows
            if bool(segment.get("passes_rules"))
        ]
        selected_covering_segment_ids = [
            segment_id
            for segment_id in segment_ids
            if segment_id in selected_segment_ids
        ]
        rule_failure_counts: dict[str, int] = {}
        for segment in segment_rows:
            for reason in cast(list[Any], segment.get("rule_failures", [])):
                if not isinstance(reason, str):
                    continue
                rule_failure_counts[reason] = rule_failure_counts.get(reason, 0) + 1
        best_segment = None
        if segment_rows:
            best_segment = max(
                segment_rows,
                key=lambda item: (
                    1 if bool(item.get("passes_rules")) else 0,
                    float(item.get("score", 0.0)),
                    float(item.get("duration_sec", 0.0)),
                    -float(item.get("combined_condition_number", float("inf"))),
                ),
            )
        if key_tuple in selected_key_tuples:
            rejection_classification = "selected_segment"
        elif passing_segment_ids:
            rejection_classification = "excluded_by_information_capacity_merge"
        elif segment_rows:
            rejection_classification = "no_rule_passing_segment"
        else:
            rejection_classification = "outside_segment_windowing"

        start_timestamp_ns, end_timestamp_ns, frame_stride = key_tuple
        decisions.append(
            {
                "start_timestamp_ns": int(start_timestamp_ns),
                "end_timestamp_ns": int(end_timestamp_ns),
                "frame_stride": int(frame_stride),
                "window_id": (
                    None
                    if candidate.get("window_id") is None
                    else int(candidate["window_id"])
                ),
                "registration_fitness": (
                    None
                    if candidate.get("registration_fitness") is None
                    else float(candidate["registration_fitness"])
                ),
                "probabilistic_information_score": (
                    None
                    if candidate.get("probabilistic_information_score") is None
                    else float(candidate["probabilistic_information_score"])
                ),
                "rejection_classification": rejection_classification,
                "segment_count": int(len(segment_rows)),
                "passing_segment_count": int(len(passing_segment_ids)),
                "passing_segment_ids": [
                    int(segment_id) for segment_id in passing_segment_ids
                ],
                "selected_covering_segment_ids": [
                    int(segment_id) for segment_id in selected_covering_segment_ids
                ],
                "rule_failure_counts": {
                    reason: int(count)
                    for reason, count in sorted(
                        rule_failure_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                },
                "best_segment_id": (
                    None if best_segment is None else int(best_segment["segment_id"])
                ),
                "best_segment_passes_rules": (
                    None
                    if best_segment is None
                    else bool(best_segment.get("passes_rules"))
                ),
                "best_segment_rule_failures": (
                    []
                    if best_segment is None
                    else [
                        str(reason)
                        for reason in cast(
                            list[Any], best_segment.get("rule_failures", [])
                        )
                        if isinstance(reason, str)
                    ]
                ),
                "best_segment_advisory_reasons": (
                    []
                    if best_segment is None
                    else [
                        str(reason)
                        for reason in cast(
                            list[Any], best_segment.get("advisory_reasons", [])
                        )
                        if isinstance(reason, str)
                    ]
                ),
                "best_segment_rotation_min_eigenvalue": (
                    None
                    if best_segment is None
                    else float(best_segment.get("rotation_min_eigenvalue", 0.0))
                ),
                "best_segment_planar_min_eigenvalue": (
                    None
                    if best_segment is None
                    else float(best_segment.get("planar_min_eigenvalue", 0.0))
                ),
                "best_segment_combined_condition_number": (
                    None
                    if best_segment is None
                    else float(
                        best_segment.get("combined_condition_number", float("inf"))
                    )
                ),
            }
        )
    return decisions


def _candidate_midpoint_timestamp_ns(candidate: dict[str, Any]) -> int:
    return int(
        (
            int(candidate["start_meta"].timestamp_ns)
            + int(candidate["end_meta"].timestamp_ns)
        )
        // 2
    )


def _candidate_duration_sec(candidate: dict[str, Any]) -> float:
    start_timestamp_ns = int(candidate["start_meta"].timestamp_ns)
    end_timestamp_ns = int(candidate["end_meta"].timestamp_ns)
    return max(float(end_timestamp_ns - start_timestamp_ns) / 1e9, 1e-3)


def _candidate_weight(candidate: dict[str, Any], key: str) -> float:
    raw_weight = candidate.get(key, candidate.get("weight", 1.0))
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(weight) or weight <= 0.0:
        return 1.0
    return float(weight)


def _normalized_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    safe = [max(float(weight), 1e-9) for weight in weights]
    mean_weight = float(sum(safe) / len(safe))
    if not math.isfinite(mean_weight) or mean_weight <= 1e-9:
        return [1.0 for _ in safe]
    return [float(weight / mean_weight) for weight in safe]


def _skew_symmetric(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -float(vector[2]), float(vector[1])],
            [float(vector[2]), 0.0, -float(vector[0])],
            [-float(vector[1]), float(vector[0]), 0.0],
        ],
        dtype=float,
    )


def _matrix_metrics(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return {
            "eigenvalues": [],
            "rank": 0,
            "condition_number": float("inf"),
            "min_eigenvalue": 0.0,
            "max_eigenvalue": 0.0,
        }
    symmetric_matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_matrix)
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size >= 2:
        condition_number = float(positive[-1] / positive[0])
    elif positive.size == 1:
        condition_number = 1.0
    else:
        condition_number = float("inf")
    return {
        "eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "rank": int(positive.size),
        "condition_number": float(condition_number),
        "min_eigenvalue": (0.0 if positive.size == 0 else float(np.min(positive))),
        "max_eigenvalue": (0.0 if positive.size == 0 else float(np.max(positive))),
    }


def _segment_fisher_metrics(
    candidates: list[dict[str, Any]],
    *,
    rotation_sigma_rad: float,
    translation_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    rotation_variance = max(float(rotation_sigma_rad) ** 2, 1e-9)
    translation_variance = max(float(translation_sigma_m) ** 2, 1e-9)
    rotation_raw_weights = [
        _candidate_weight(candidate, "information_rotation_confidence")
        for candidate in candidates
    ]
    translation_raw_weights = [
        _candidate_weight(candidate, "information_translation_confidence")
        for candidate in candidates
    ]
    rotation_weights = _normalized_weights(rotation_raw_weights)
    translation_weights = _normalized_weights(translation_raw_weights)

    rotation_fisher = np.zeros((3, 3), dtype=float)
    planar_fisher_xy = np.zeros((2, 2), dtype=float)
    yaw_information = 0.0

    for candidate, rotation_weight, translation_weight in zip(
        candidates,
        rotation_weights,
        translation_weights,
    ):
        imu_delta = np.asarray(candidate["imu_delta"], dtype=float)
        imu_rotation = np.asarray(imu_delta[:3, :3], dtype=float)
        duration_sec = _candidate_duration_sec(candidate)
        rotvec = R.from_matrix(imu_rotation).as_rotvec()
        angular_rate = np.asarray(rotvec, dtype=float) / float(duration_sec)

        rotation_design = _skew_symmetric(angular_rate)
        rotation_precision = float(rotation_weight) / float(rotation_variance)
        rotation_fisher += rotation_precision * (rotation_design.T @ rotation_design)

        planar_design_xy = np.asarray(
            (imu_rotation - np.eye(3, dtype=float))[:, :2],
            dtype=float,
        )
        translation_precision = float(translation_weight) / float(translation_variance)
        planar_fisher_xy += translation_precision * (
            planar_design_xy.T @ planar_design_xy
        )

        yaw_rad = float(
            np.radians(
                float(
                    candidate.get(
                        "imu_signed_yaw_deg",
                        0.0,
                    )
                )
            )
        )
        yaw_information += float(rotation_precision) * float(yaw_rad * yaw_rad)

    return rotation_fisher, planar_fisher_xy, float(yaw_information)


def _segment_information_matrix(
    rotation_fisher_matrix: np.ndarray,
    planar_fisher_xy_matrix: np.ndarray,
    yaw_information: float,
) -> np.ndarray:
    information_matrix = np.zeros((6, 6), dtype=float)
    information_matrix[:3, :3] = np.asarray(rotation_fisher_matrix, dtype=float)
    information_matrix[3:5, 3:5] = np.asarray(planar_fisher_xy_matrix, dtype=float)
    information_matrix[5, 5] = max(float(yaw_information), 0.0)
    return 0.5 * (information_matrix + information_matrix.T)


def _segment_matrix_from_summary(segment: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(segment.get("information_matrix", []), dtype=float)
    if matrix.shape != (6, 6):
        return np.zeros((6, 6), dtype=float)
    return 0.5 * (matrix + matrix.T)


def _safe_condition_number(condition_number: float) -> float:
    if not math.isfinite(float(condition_number)):
        return float("inf")
    return max(float(condition_number), 1.0)


def _candidate_observability_score(candidate: dict[str, Any]) -> float:
    for key in (
        "probabilistic_information_score",
        "information_score",
        "score",
        "weight",
    ):
        raw_value = candidate.get(key)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return float(value)
    return 0.0


def _candidate_registration_fitness(candidate: dict[str, Any]) -> float:
    raw_value = candidate.get("registration_fitness")
    if raw_value is None:
        return 0.0
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(float(value), 0.0)


def _window_based_rescue_candidates(
    *,
    sorted_candidates: list[dict[str, Any]],
    selected_key_tuples: set[tuple[int, int, int]],
    candidate_decisions: list[dict[str, Any]],
    max_combined_condition_number: float,
    max_selected_candidates: int | None,
    auto_window_rescue_count: int,
    auto_window_rescue_min_relative_score: float,
    auto_window_rescue_condition_scale: float,
) -> dict[str, Any]:
    rescue_summary: dict[str, Any] = {
        "enabled": bool(auto_window_rescue_count > 0),
        "requested_window_count": int(max(auto_window_rescue_count, 0)),
        "applied_count": 0,
        "rescued_candidate_keys": [],
        "rescued_window_ids": [],
        "rescued_windows": [],
        "selection_anchor_mean_score": 0.0,
        "selection_anchor_mean_fitness": 0.0,
        "min_relative_score": float(max(auto_window_rescue_min_relative_score, 0.0)),
        "condition_scale": float(max(auto_window_rescue_condition_scale, 1.0)),
    }
    if auto_window_rescue_count <= 0:
        rescue_summary["status"] = "disabled"
        return rescue_summary

    selected_candidates = [
        candidate
        for candidate in sorted_candidates
        if registered_candidate_key(candidate) in selected_key_tuples
    ]
    if not selected_candidates:
        rescue_summary["status"] = "no_selected_anchor_candidates"
        return rescue_summary

    selected_scores = [
        _candidate_observability_score(candidate) for candidate in selected_candidates
    ]
    selected_scores = [score for score in selected_scores if score > 0.0]
    selected_fitnesses = [
        _candidate_registration_fitness(candidate) for candidate in selected_candidates
    ]
    selected_fitnesses = [fitness for fitness in selected_fitnesses if fitness > 0.0]
    anchor_mean_score = (
        float(np.mean(np.asarray(selected_scores, dtype=float)))
        if selected_scores
        else 1.0
    )
    anchor_mean_fitness = (
        float(np.mean(np.asarray(selected_fitnesses, dtype=float)))
        if selected_fitnesses
        else 1.0
    )
    rescue_summary["selection_anchor_mean_score"] = float(anchor_mean_score)
    rescue_summary["selection_anchor_mean_fitness"] = float(anchor_mean_fitness)

    decision_by_key = {
        (
            int(decision["start_timestamp_ns"]),
            int(decision["end_timestamp_ns"]),
            int(decision["frame_stride"]),
        ): decision
        for decision in candidate_decisions
        if isinstance(decision, dict)
        and "start_timestamp_ns" in decision
        and "end_timestamp_ns" in decision
        and "frame_stride" in decision
    }
    outside_by_window: dict[int, list[dict[str, Any]]] = {}
    for candidate in sorted_candidates:
        key_tuple = registered_candidate_key(candidate)
        if key_tuple in selected_key_tuples:
            continue
        window_id_raw = candidate.get("window_id")
        if window_id_raw is None:
            continue
        try:
            window_id = int(window_id_raw)
        except (TypeError, ValueError):
            continue
        decision = decision_by_key.get(key_tuple, {})
        outside_by_window.setdefault(window_id, []).append(
            {
                "candidate": candidate,
                "candidate_key": key_tuple,
                "decision": decision,
            }
        )
    if not outside_by_window:
        rescue_summary["status"] = "no_outside_candidates"
        return rescue_summary

    condition_limit = max(float(max_combined_condition_number), 1.0) * max(
        float(auto_window_rescue_condition_scale), 1.0
    )
    min_relative_score = max(float(auto_window_rescue_min_relative_score), 0.0)
    blocking_rule_failures = {
        "rotation_lambda_below_threshold",
        "planar_lambda_below_threshold",
    }
    window_rows: list[dict[str, Any]] = []
    for window_id, rows in outside_by_window.items():
        candidate_scores = [
            _candidate_observability_score(cast(dict[str, Any], row["candidate"]))
            for row in rows
        ]
        candidate_fitnesses = [
            _candidate_registration_fitness(cast(dict[str, Any], row["candidate"]))
            for row in rows
        ]
        mean_score = float(np.mean(np.asarray(candidate_scores, dtype=float)))
        mean_fitness = float(np.mean(np.asarray(candidate_fitnesses, dtype=float)))
        relative_score = (
            float(mean_score / max(anchor_mean_score, 1e-9))
            if anchor_mean_score > 0.0
            else 1.0
        )
        relative_fitness = (
            float(mean_fitness / max(anchor_mean_fitness, 1e-9))
            if anchor_mean_fitness > 0.0
            else 1.0
        )
        rule_failure_counts: dict[str, int] = {}
        rejection_class_counts: dict[str, int] = {}
        best_segment_conditions: list[float] = []
        for row in rows:
            decision = cast(dict[str, Any], row.get("decision") or {})
            rejection_class = str(
                decision.get("rejection_classification") or "outside_segment_windowing"
            )
            rejection_class_counts[rejection_class] = (
                rejection_class_counts.get(rejection_class, 0) + 1
            )
            for reason, count in cast(
                dict[str, Any], decision.get("rule_failure_counts") or {}
            ).items():
                try:
                    parsed_count = int(count)
                except (TypeError, ValueError):
                    continue
                rule_failure_counts[str(reason)] = (
                    rule_failure_counts.get(str(reason), 0) + parsed_count
                )
            best_condition_raw = decision.get("best_segment_combined_condition_number")
            if best_condition_raw is None:
                continue
            try:
                best_condition_value = float(best_condition_raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(best_condition_value):
                best_segment_conditions.append(best_condition_value)
        mean_best_segment_condition = (
            float(np.mean(np.asarray(best_segment_conditions, dtype=float)))
            if best_segment_conditions
            else float("inf")
        )
        has_blocking_failures = any(
            rule_failure_counts.get(reason, 0) > 0 for reason in blocking_rule_failures
        )
        condition_ok = (
            True
            if not math.isfinite(mean_best_segment_condition)
            else bool(mean_best_segment_condition <= condition_limit)
        )
        eligible = bool(
            relative_score >= min_relative_score
            and condition_ok
            and not has_blocking_failures
        )
        best_row = max(
            rows,
            key=lambda item: (
                _candidate_observability_score(cast(dict[str, Any], item["candidate"])),
                _candidate_registration_fitness(
                    cast(dict[str, Any], item["candidate"])
                ),
            ),
        )
        best_key = cast(tuple[int, int, int], best_row["candidate_key"])
        window_rows.append(
            {
                "window_id": int(window_id),
                "candidate_count": int(len(rows)),
                "mean_observability_score": float(mean_score),
                "mean_registration_fitness": float(mean_fitness),
                "relative_observability_score": float(relative_score),
                "relative_registration_fitness": float(relative_fitness),
                "mean_best_segment_combined_condition_number": (
                    None
                    if not math.isfinite(mean_best_segment_condition)
                    else float(mean_best_segment_condition)
                ),
                "condition_limit": float(condition_limit),
                "condition_ok": bool(condition_ok),
                "blocking_failures": sorted(
                    [
                        reason
                        for reason in blocking_rule_failures
                        if rule_failure_counts.get(reason, 0) > 0
                    ]
                ),
                "rule_failure_counts": {
                    reason: int(count)
                    for reason, count in sorted(
                        rule_failure_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                },
                "rejection_class_counts": {
                    reason: int(count)
                    for reason, count in sorted(
                        rejection_class_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                },
                "eligible": bool(eligible),
                "best_candidate_key": {
                    "start_timestamp_ns": int(best_key[0]),
                    "end_timestamp_ns": int(best_key[1]),
                    "frame_stride": int(best_key[2]),
                },
                "priority_score": float(
                    0.7 * relative_score
                    + 0.2 * relative_fitness
                    + (
                        0.1
                        if rejection_class_counts.get("outside_segment_windowing", 0)
                        > 0
                        else 0.0
                    )
                ),
            }
        )

    rescue_summary["window_summaries"] = sorted(
        window_rows,
        key=lambda item: (
            0 if bool(item["eligible"]) else 1,
            -float(item["priority_score"]),
            -float(item["relative_observability_score"]),
            -float(item["mean_registration_fitness"]),
        ),
    )
    rescued_windows = []
    rescued_key_payloads = []
    for window_row in cast(list[dict[str, Any]], rescue_summary["window_summaries"]):
        if len(rescued_windows) >= int(auto_window_rescue_count):
            break
        if not bool(window_row["eligible"]):
            continue
        best_key_payload = cast(dict[str, Any], window_row["best_candidate_key"])
        key_tuple = (
            int(best_key_payload["start_timestamp_ns"]),
            int(best_key_payload["end_timestamp_ns"]),
            int(best_key_payload["frame_stride"]),
        )
        if key_tuple in selected_key_tuples:
            continue
        if max_selected_candidates is not None and len(selected_key_tuples) >= int(
            max_selected_candidates
        ):
            break
        selected_key_tuples.add(key_tuple)
        rescued_windows.append(int(window_row["window_id"]))
        rescued_key_payloads.append(
            {
                "start_timestamp_ns": int(key_tuple[0]),
                "end_timestamp_ns": int(key_tuple[1]),
                "frame_stride": int(key_tuple[2]),
            }
        )
        window_row["rescued"] = True
    rescue_summary["rescued_candidate_keys"] = _sort_candidate_key_payloads(
        rescued_key_payloads
    )
    rescue_summary["rescued_window_ids"] = sorted(set(rescued_windows))
    rescue_summary["rescued_windows"] = [
        row
        for row in cast(list[dict[str, Any]], rescue_summary["window_summaries"])
        if bool(row.get("rescued"))
    ]
    rescue_summary["applied_count"] = int(len(rescue_summary["rescued_candidate_keys"]))
    rescue_summary["status"] = (
        "applied"
        if rescue_summary["applied_count"] > 0
        else "no_eligible_window_for_rescue"
    )
    return rescue_summary


def _segment_summary(
    *,
    segment_id: int,
    candidates: list[dict[str, Any]],
    target_window_sec: float,
    min_window_sec: float,
    min_samples: int,
    min_rotation_lambda: float,
    min_planar_lambda: float,
    max_combined_condition_number: float,
    turn_condition_relax_scale: float,
    straight_condition_strict_scale: float,
    turn_segment_min_turn_ratio: float,
    turn_segment_yaw_p95_deg: float,
    straight_segment_max_turn_ratio: float,
    straight_segment_yaw_p95_deg: float,
    rotation_sigma_rad: float,
    translation_sigma_m: float,
    min_turn_balance_ratio: float,
    min_heading_bin_count: int,
) -> dict[str, Any]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: int(item["start_meta"].timestamp_ns),
    )
    start_timestamp_ns = int(sorted_candidates[0]["start_meta"].timestamp_ns)
    end_timestamp_ns = int(sorted_candidates[-1]["end_meta"].timestamp_ns)
    duration_sec = max(float(end_timestamp_ns - start_timestamp_ns) / 1e9, 0.0)
    window_ids = sorted(
        {
            int(candidate["window_id"])
            for candidate in sorted_candidates
            if candidate.get("window_id") is not None
        }
    )

    turn_counts = {"left": 0, "right": 0, "neutral": 0}
    heading_bins = set()
    yaw_abs_values_deg: list[float] = []
    for candidate in sorted_candidates:
        yaw_abs_deg = abs(float(candidate.get("imu_signed_yaw_deg", 0.0)))
        yaw_abs_values_deg.append(float(yaw_abs_deg))
        turn_sign = motion_turn_sign(float(candidate.get("imu_signed_yaw_deg", 0.0)))
        if turn_sign in ("left", "right"):
            turn_counts[turn_sign] += 1
        else:
            turn_counts["neutral"] += 1
        heading_index = heading_bin(candidate.get("imu_translation_heading_deg"))
        if heading_index is not None:
            heading_bins.add(int(heading_index))
    active_turn_count = int(turn_counts["left"] + turn_counts["right"])
    if active_turn_count > 0:
        turn_balance_ratio = float(
            min(turn_counts["left"], turn_counts["right"]) / active_turn_count
        )
    else:
        turn_balance_ratio = 0.0
    heading_bin_count = int(len(heading_bins))
    turn_sample_ratio = float(active_turn_count / max(len(sorted_candidates), 1))
    yaw_abs_p95_deg = (
        float(np.percentile(np.asarray(yaw_abs_values_deg, dtype=float), 95))
        if yaw_abs_values_deg
        else 0.0
    )
    motion_regime = "mixed"
    condition_threshold_scale = 1.0
    if turn_sample_ratio >= float(
        turn_segment_min_turn_ratio
    ) or yaw_abs_p95_deg >= float(turn_segment_yaw_p95_deg):
        motion_regime = "turn_dominant"
        condition_threshold_scale = max(float(turn_condition_relax_scale), 1.0)
    elif turn_sample_ratio <= float(
        straight_segment_max_turn_ratio
    ) and yaw_abs_p95_deg <= float(straight_segment_yaw_p95_deg):
        motion_regime = "straight_dominant"
        condition_threshold_scale = min(
            max(float(straight_condition_strict_scale), 1e-3), 1.0
        )
    effective_max_combined_condition_number = max(
        float(max_combined_condition_number) * float(condition_threshold_scale),
        1.0,
    )

    (
        rotation_fisher_matrix,
        planar_fisher_xy_matrix,
        yaw_information,
    ) = _segment_fisher_metrics(
        sorted_candidates,
        rotation_sigma_rad=rotation_sigma_rad,
        translation_sigma_m=translation_sigma_m,
    )
    rotation_fisher = _matrix_metrics(rotation_fisher_matrix)
    planar_fisher_xy = _matrix_metrics(planar_fisher_xy_matrix)
    information_matrix = _segment_information_matrix(
        rotation_fisher_matrix,
        planar_fisher_xy_matrix,
        yaw_information,
    )
    information_metrics = _matrix_metrics(information_matrix)
    rotation_condition_number = float(rotation_fisher["condition_number"])
    planar_condition_number = float(planar_fisher_xy["condition_number"])
    rotation_min_eigenvalue = float(rotation_fisher["min_eigenvalue"])
    planar_min_eigenvalue = float(planar_fisher_xy["min_eigenvalue"])
    combined_condition_number = float(
        max(rotation_condition_number, planar_condition_number)
    )
    combined_min_eigenvalue = float(min(rotation_min_eigenvalue, planar_min_eigenvalue))

    duration_closeness = max(
        0.0,
        1.0
        - (
            abs(float(duration_sec - target_window_sec))
            / max(float(target_window_sec), 1e-6)
        ),
    )
    heading_diversity = min(float(heading_bin_count) / 3.0, 1.0)
    conditioning_term = 0.0
    if math.isfinite(combined_condition_number):
        conditioning_term = math.log1p(max(combined_condition_number, 1.0))
    else:
        conditioning_term = math.log1p(1e9)
    observability_gain = (
        math.log1p(max(rotation_min_eigenvalue, 0.0) * 1e3)
        + math.log1p(max(planar_min_eigenvalue, 0.0) * 1e2)
        + 0.5 * math.log1p(max(yaw_information, 0.0) * 20.0)
    )
    score = (
        1.2 * float(observability_gain)
        + 0.6 * float(heading_diversity)
        + 0.3 * float(duration_closeness)
        - 0.35 * float(conditioning_term)
    )

    rule_failures = []
    advisory_reasons = []
    if int(len(sorted_candidates)) < int(min_samples):
        rule_failures.append("insufficient_samples")
    if float(duration_sec) < float(min_window_sec):
        rule_failures.append("segment_too_short")
    if rotation_min_eigenvalue < float(min_rotation_lambda):
        rule_failures.append("rotation_lambda_below_threshold")
    if planar_min_eigenvalue < float(min_planar_lambda):
        rule_failures.append("planar_lambda_below_threshold")
    if not math.isfinite(
        combined_condition_number
    ) or combined_condition_number > float(effective_max_combined_condition_number):
        rule_failures.append("combined_condition_too_high")
    if float(turn_balance_ratio) < float(min_turn_balance_ratio):
        advisory_reasons.append("turn_balance_too_low")
    if int(heading_bin_count) < int(min_heading_bin_count):
        advisory_reasons.append("heading_bin_count_too_low")

    return {
        "segment_id": int(segment_id),
        "start_timestamp_ns": int(start_timestamp_ns),
        "end_timestamp_ns": int(end_timestamp_ns),
        "duration_sec": float(duration_sec),
        "sample_count": int(len(sorted_candidates)),
        "window_ids": [int(window_id) for window_id in window_ids],
        "turn_counts": {
            "left": int(turn_counts["left"]),
            "right": int(turn_counts["right"]),
            "neutral": int(turn_counts["neutral"]),
        },
        "turn_balance_ratio": float(turn_balance_ratio),
        "turn_sample_ratio": float(turn_sample_ratio),
        "yaw_abs_p95_deg": float(yaw_abs_p95_deg),
        "motion_regime": str(motion_regime),
        "condition_threshold_scale": float(condition_threshold_scale),
        "effective_max_combined_condition_number": float(
            effective_max_combined_condition_number
        ),
        "heading_bin_count": int(heading_bin_count),
        "rotation_min_eigenvalue": float(rotation_min_eigenvalue),
        "planar_min_eigenvalue": float(planar_min_eigenvalue),
        "combined_min_eigenvalue": float(combined_min_eigenvalue),
        "rotation_condition_number": float(rotation_condition_number),
        "planar_condition_number": float(planar_condition_number),
        "combined_condition_number": float(combined_condition_number),
        "information_matrix": information_matrix.tolist(),
        "information_metrics": information_metrics,
        "rotation_fisher": rotation_fisher,
        "planar_fisher_xy": planar_fisher_xy,
        "yaw_information": float(yaw_information),
        "score": float(score),
        "passes_rules": not rule_failures,
        "rule_failures": list(rule_failures),
        "advisory_reasons": list(advisory_reasons),
        "candidate_keys": [
            _registered_candidate_key_payload(candidate)
            for candidate in sorted_candidates
        ],
    }


def screen_motion_candidates_by_gril_observability(
    motion_registered_candidates: list[dict[str, Any]],
    *,
    mode: str,
    target_window_sec: float,
    min_window_sec: float,
    min_samples: int,
    top_segments: int,
    min_rotation_lambda: float = 1e-4,
    min_planar_lambda: float = 1e-3,
    max_combined_condition_number: float = 2e3,
    turn_condition_relax_scale: float = 2.0,
    straight_condition_strict_scale: float = 0.8,
    turn_segment_min_turn_ratio: float = 0.35,
    turn_segment_yaw_p95_deg: float = 8.0,
    straight_segment_max_turn_ratio: float = 0.15,
    straight_segment_yaw_p95_deg: float = 3.0,
    rotation_sigma_rad: float = 0.02,
    translation_sigma_m: float = 0.05,
    max_merged_segments: int = 2,
    max_selected_candidates: int | None = None,
    auto_window_rescue_count: int = 0,
    auto_window_rescue_min_relative_score: float = 0.65,
    auto_window_rescue_condition_scale: float = 1.5,
) -> dict[str, Any]:
    enabled = str(mode) != "off"
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": str(mode),
        "status": "disabled" if not enabled else "pending",
        "applied": False,
        "target_window_sec": float(target_window_sec),
        "min_window_sec": float(min_window_sec),
        "min_samples": int(max(min_samples, 1)),
        "top_segments": int(max(top_segments, 1)),
        "max_merged_segments": int(max(max_merged_segments, 1)),
        "max_selected_candidates": (
            None
            if max_selected_candidates is None
            else int(max(max_selected_candidates, 1))
        ),
        "segment_count": 0,
        "selected_candidate_count": 0,
        "selected_window_ids": [],
        "selected_candidate_keys": [],
        "selected_segment": None,
        "selected_segments": [],
        "best_segment": None,
        "thresholds": {
            "min_rotation_lambda": float(min_rotation_lambda),
            "min_planar_lambda": float(min_planar_lambda),
            "max_combined_condition_number": float(max_combined_condition_number),
            "turn_condition_relax_scale": float(turn_condition_relax_scale),
            "straight_condition_strict_scale": float(straight_condition_strict_scale),
            "turn_segment_min_turn_ratio": float(turn_segment_min_turn_ratio),
            "turn_segment_yaw_p95_deg": float(turn_segment_yaw_p95_deg),
            "straight_segment_max_turn_ratio": float(straight_segment_max_turn_ratio),
            "straight_segment_yaw_p95_deg": float(straight_segment_yaw_p95_deg),
            "rotation_sigma_rad": float(rotation_sigma_rad),
            "translation_sigma_m": float(translation_sigma_m),
            "min_turn_balance_ratio": 0.10,
            "min_heading_bin_count": 1,
            "auto_window_rescue_count": int(max(auto_window_rescue_count, 0)),
            "auto_window_rescue_min_relative_score": float(
                max(auto_window_rescue_min_relative_score, 0.0)
            ),
            "auto_window_rescue_condition_scale": float(
                max(auto_window_rescue_condition_scale, 1.0)
            ),
        },
        "segments": [],
        "candidate_decisions": [],
        "auto_window_rescue": {
            "enabled": bool(auto_window_rescue_count > 0),
            "status": "pending" if auto_window_rescue_count > 0 else "disabled",
            "requested_window_count": int(max(auto_window_rescue_count, 0)),
        },
    }
    if not enabled:
        return summary
    if str(mode) != "gril_fisher":
        summary["status"] = "unsupported_mode"
        return summary
    if not motion_registered_candidates:
        summary["status"] = "no_registered_candidates"
        return summary

    target_window_sec = max(float(target_window_sec), 1.0)
    min_window_sec = min(max(float(min_window_sec), 0.1), float(target_window_sec))
    min_samples = max(int(min_samples), 1)
    top_segments = max(int(top_segments), 1)
    max_merged_segments = max(int(max_merged_segments), 1)
    max_selected_candidates = (
        None
        if max_selected_candidates is None
        else max(int(max_selected_candidates), 1)
    )
    min_rotation_lambda = max(float(min_rotation_lambda), 0.0)
    min_planar_lambda = max(float(min_planar_lambda), 0.0)
    max_combined_condition_number = max(float(max_combined_condition_number), 1.0)
    turn_condition_relax_scale = max(float(turn_condition_relax_scale), 1.0)
    straight_condition_strict_scale = min(
        max(float(straight_condition_strict_scale), 1e-3),
        1.0,
    )
    turn_segment_min_turn_ratio = min(max(float(turn_segment_min_turn_ratio), 0.0), 1.0)
    turn_segment_yaw_p95_deg = max(float(turn_segment_yaw_p95_deg), 0.0)
    straight_segment_max_turn_ratio = min(
        max(float(straight_segment_max_turn_ratio), 0.0),
        1.0,
    )
    straight_segment_yaw_p95_deg = max(float(straight_segment_yaw_p95_deg), 0.0)
    rotation_sigma_rad = max(float(rotation_sigma_rad), 1e-4)
    translation_sigma_m = max(float(translation_sigma_m), 1e-4)
    auto_window_rescue_count = max(int(auto_window_rescue_count), 0)
    auto_window_rescue_min_relative_score = max(
        float(auto_window_rescue_min_relative_score),
        0.0,
    )
    auto_window_rescue_condition_scale = max(
        float(auto_window_rescue_condition_scale),
        1.0,
    )
    capacity_min_gain_ratio = 0.08
    capacity_min_gain_abs = 1e-4
    capacity_max_condition_worsening_ratio = 3.0
    summary["target_window_sec"] = float(target_window_sec)
    summary["min_window_sec"] = float(min_window_sec)
    summary["min_samples"] = int(min_samples)
    summary["top_segments"] = int(top_segments)
    summary["max_merged_segments"] = int(max_merged_segments)
    summary["max_selected_candidates"] = (
        None if max_selected_candidates is None else int(max_selected_candidates)
    )
    summary["thresholds"] = {
        "min_rotation_lambda": float(min_rotation_lambda),
        "min_planar_lambda": float(min_planar_lambda),
        "max_combined_condition_number": float(max_combined_condition_number),
        "turn_condition_relax_scale": float(turn_condition_relax_scale),
        "straight_condition_strict_scale": float(straight_condition_strict_scale),
        "turn_segment_min_turn_ratio": float(turn_segment_min_turn_ratio),
        "turn_segment_yaw_p95_deg": float(turn_segment_yaw_p95_deg),
        "straight_segment_max_turn_ratio": float(straight_segment_max_turn_ratio),
        "straight_segment_yaw_p95_deg": float(straight_segment_yaw_p95_deg),
        "rotation_sigma_rad": float(rotation_sigma_rad),
        "translation_sigma_m": float(translation_sigma_m),
        "min_turn_balance_ratio": float(
            summary["thresholds"]["min_turn_balance_ratio"]
        ),
        "min_heading_bin_count": int(summary["thresholds"]["min_heading_bin_count"]),
        "auto_window_rescue_count": int(auto_window_rescue_count),
        "auto_window_rescue_min_relative_score": float(
            auto_window_rescue_min_relative_score
        ),
        "auto_window_rescue_condition_scale": float(auto_window_rescue_condition_scale),
        "capacity_min_gain_ratio": float(capacity_min_gain_ratio),
        "capacity_min_gain_abs": float(capacity_min_gain_abs),
        "capacity_max_condition_worsening_ratio": float(
            capacity_max_condition_worsening_ratio
        ),
    }

    sorted_candidates = sorted(
        motion_registered_candidates,
        key=lambda item: _candidate_midpoint_timestamp_ns(item),
    )
    candidates_by_key = {
        registered_candidate_key(candidate): candidate
        for candidate in sorted_candidates
    }
    midpoint_timestamps = [
        _candidate_midpoint_timestamp_ns(candidate) for candidate in sorted_candidates
    ]
    target_window_ns = int(round(target_window_sec * 1e9))
    segment_summaries = []
    end_index = 0
    segment_id = 0
    for start_index in range(len(sorted_candidates)):
        if end_index < start_index + 1:
            end_index = start_index + 1
        while (
            end_index < len(sorted_candidates)
            and midpoint_timestamps[end_index] - midpoint_timestamps[start_index]
            <= target_window_ns
        ):
            end_index += 1
        segment_candidates = sorted_candidates[start_index:end_index]
        if len(segment_candidates) < min_samples:
            continue
        segment_summaries.append(
            _segment_summary(
                segment_id=segment_id,
                candidates=segment_candidates,
                target_window_sec=target_window_sec,
                min_window_sec=min_window_sec,
                min_samples=min_samples,
                min_rotation_lambda=min_rotation_lambda,
                min_planar_lambda=min_planar_lambda,
                max_combined_condition_number=max_combined_condition_number,
                turn_condition_relax_scale=turn_condition_relax_scale,
                straight_condition_strict_scale=straight_condition_strict_scale,
                turn_segment_min_turn_ratio=turn_segment_min_turn_ratio,
                turn_segment_yaw_p95_deg=turn_segment_yaw_p95_deg,
                straight_segment_max_turn_ratio=straight_segment_max_turn_ratio,
                straight_segment_yaw_p95_deg=straight_segment_yaw_p95_deg,
                rotation_sigma_rad=rotation_sigma_rad,
                translation_sigma_m=translation_sigma_m,
                min_turn_balance_ratio=float(
                    summary["thresholds"]["min_turn_balance_ratio"]
                ),
                min_heading_bin_count=int(
                    summary["thresholds"]["min_heading_bin_count"]
                ),
            )
        )
        segment_id += 1

    if not segment_summaries:
        summary["status"] = "no_segment_with_min_samples"
        summary["candidate_decisions"] = _candidate_screening_decisions(
            candidates_by_key=candidates_by_key,
            segment_summaries=[],
            selected_segments=[],
            selected_key_tuples=set(),
        )
        return summary

    ranked_segments = sorted(
        segment_summaries,
        key=lambda item: (
            0 if bool(item["passes_rules"]) else 1,
            (
                float(item["combined_condition_number"])
                if math.isfinite(float(item["combined_condition_number"]))
                else float("inf")
            ),
            -float(item["combined_min_eigenvalue"]),
            -float(item["score"]),
            -float(item["duration_sec"]),
        ),
    )
    summary["segment_count"] = int(len(ranked_segments))
    summary["segments"] = ranked_segments[:top_segments]
    summary["best_segment"] = ranked_segments[0]

    passing_segments = [
        segment for segment in ranked_segments if bool(segment["passes_rules"])
    ]
    if not passing_segments:
        summary["status"] = "no_rule_passing_segment"
        summary["candidate_decisions"] = _candidate_screening_decisions(
            candidates_by_key=candidates_by_key,
            segment_summaries=ranked_segments,
            selected_segments=[],
            selected_key_tuples=set(),
        )
        return summary

    selected_segments: list[dict[str, Any]] = []
    selected_segment_traces: list[dict[str, Any]] = []
    selected_key_payloads: list[dict[str, int]] = []
    selected_key_tuples: set[tuple[int, int, int]] = set()
    running_information_matrix = np.zeros((6, 6), dtype=float)
    running_information_metrics = _matrix_metrics(running_information_matrix)
    remaining_segments = list(passing_segments)
    while remaining_segments and len(selected_segments) < max_merged_segments:
        best_choice = None
        for segment in remaining_segments:
            segment_matrix = _segment_matrix_from_summary(segment)
            candidate_information_matrix = running_information_matrix + segment_matrix
            candidate_information_metrics = _matrix_metrics(
                candidate_information_matrix
            )

            previous_min_eigenvalue = float(
                running_information_metrics.get("min_eigenvalue", 0.0)
            )
            previous_condition_number = _safe_condition_number(
                float(running_information_metrics.get("condition_number", float("inf")))
            )
            candidate_min_eigenvalue = float(
                candidate_information_metrics.get("min_eigenvalue", 0.0)
            )
            candidate_condition_number = _safe_condition_number(
                float(
                    candidate_information_metrics.get("condition_number", float("inf"))
                )
            )
            min_eigenvalue_gain = max(
                candidate_min_eigenvalue - previous_min_eigenvalue,
                0.0,
            )
            min_eigenvalue_gain_ratio = (
                float(min_eigenvalue_gain / previous_min_eigenvalue)
                if previous_min_eigenvalue > 1e-9
                else (1.0 if candidate_min_eigenvalue > 0.0 else 0.0)
            )
            condition_worsening_ratio = (
                1.0
                if not math.isfinite(previous_condition_number)
                else float(
                    candidate_condition_number / max(previous_condition_number, 1.0)
                )
            )
            gain_ok = (
                min_eigenvalue_gain >= capacity_min_gain_abs
                or min_eigenvalue_gain_ratio >= capacity_min_gain_ratio
            )
            condition_ok = (
                condition_worsening_ratio <= capacity_max_condition_worsening_ratio
                or not math.isfinite(previous_condition_number)
            )
            if selected_segments and (not gain_ok or not condition_ok):
                continue

            objective = (
                float(candidate_min_eigenvalue),
                -float(math.log1p(candidate_condition_number)),
                float(min_eigenvalue_gain),
                float(min_eigenvalue_gain_ratio),
                float(segment["score"]),
            )
            if best_choice is None or objective > best_choice["objective"]:
                best_choice = {
                    "segment": segment,
                    "objective": objective,
                    "candidate_information_matrix": candidate_information_matrix,
                    "candidate_information_metrics": candidate_information_metrics,
                    "min_eigenvalue_gain": min_eigenvalue_gain,
                    "min_eigenvalue_gain_ratio": min_eigenvalue_gain_ratio,
                    "condition_worsening_ratio": condition_worsening_ratio,
                }
        if best_choice is None:
            break
        segment = cast(dict[str, Any], best_choice["segment"])
        segment_key_payloads = _sort_candidate_key_payloads(
            [
                cast(dict[str, Any], payload)
                for payload in cast(list[Any], segment["candidate_keys"])
            ]
        )
        segment_new_payloads: list[dict[str, int]] = []
        for payload in segment_key_payloads:
            key_tuple = _candidate_key_from_payload(payload)
            if key_tuple in selected_key_tuples:
                continue
            segment_new_payloads.append(payload)
        remaining_segments = [
            candidate_segment
            for candidate_segment in remaining_segments
            if int(candidate_segment["segment_id"]) != int(segment["segment_id"])
        ]
        if not segment_new_payloads:
            continue
        selected_segments.append(segment)
        selected_segment_traces.append(
            {
                "segment_id": int(segment["segment_id"]),
                "min_eigenvalue_gain": float(best_choice["min_eigenvalue_gain"]),
                "min_eigenvalue_gain_ratio": float(
                    best_choice["min_eigenvalue_gain_ratio"]
                ),
                "condition_worsening_ratio": float(
                    best_choice["condition_worsening_ratio"]
                ),
                "selected_candidate_count": int(len(segment_new_payloads)),
                "information_metrics_after_merge": cast(
                    dict[str, Any], best_choice["candidate_information_metrics"]
                ),
            }
        )
        running_information_matrix = np.asarray(
            best_choice["candidate_information_matrix"], dtype=float
        )
        running_information_metrics = cast(
            dict[str, Any], best_choice["candidate_information_metrics"]
        )
        for payload in segment_new_payloads:
            key_tuple = _candidate_key_from_payload(payload)
            selected_key_tuples.add(key_tuple)
            selected_key_payloads.append(payload)
            if (
                max_selected_candidates is not None
                and len(selected_key_payloads) >= max_selected_candidates
            ):
                break
        if (
            max_selected_candidates is not None
            and len(selected_key_payloads) >= max_selected_candidates
        ):
            break

    if not selected_segments or not selected_key_payloads:
        summary["status"] = "no_effective_segment_selection"
        summary["auto_window_rescue"] = {
            **cast(dict[str, Any], summary["auto_window_rescue"]),
            "status": "skipped_no_effective_segment_selection",
            "applied_count": 0,
            "rescued_candidate_keys": [],
            "rescued_window_ids": [],
            "rescued_windows": [],
        }
        summary["candidate_decisions"] = _candidate_screening_decisions(
            candidates_by_key=candidates_by_key,
            segment_summaries=ranked_segments,
            selected_segments=selected_segments,
            selected_key_tuples=set(),
        )
        return summary

    initial_candidate_decisions = _candidate_screening_decisions(
        candidates_by_key=candidates_by_key,
        segment_summaries=ranked_segments,
        selected_segments=selected_segments,
        selected_key_tuples=set(selected_key_tuples),
    )
    auto_window_rescue_summary = _window_based_rescue_candidates(
        sorted_candidates=sorted_candidates,
        selected_key_tuples=selected_key_tuples,
        candidate_decisions=initial_candidate_decisions,
        max_combined_condition_number=max_combined_condition_number,
        max_selected_candidates=max_selected_candidates,
        auto_window_rescue_count=auto_window_rescue_count,
        auto_window_rescue_min_relative_score=auto_window_rescue_min_relative_score,
        auto_window_rescue_condition_scale=auto_window_rescue_condition_scale,
    )
    summary["auto_window_rescue"] = auto_window_rescue_summary
    rescued_key_payloads = [
        cast(dict[str, int], payload)
        for payload in cast(
            list[Any],
            auto_window_rescue_summary.get("rescued_candidate_keys", []),
        )
        if isinstance(payload, dict)
    ]
    rescued_key_tuples = {
        _candidate_key_from_payload(payload) for payload in rescued_key_payloads
    }
    selected_payload_keys = {
        _candidate_key_from_payload(existing_payload)
        for existing_payload in selected_key_payloads
    }
    for payload in rescued_key_payloads:
        key_tuple = _candidate_key_from_payload(payload)
        if key_tuple in selected_payload_keys:
            continue
        selected_key_payloads.append(payload)
        selected_payload_keys.add(key_tuple)

    selected_window_ids = sorted(
        {
            int(candidate["window_id"])
            for candidate in sorted_candidates
            if candidate.get("window_id") is not None
            and registered_candidate_key(candidate) in selected_key_tuples
        }
    )
    selected_segment = selected_segments[0]
    summary["status"] = "applied_merged" if len(selected_segments) > 1 else "applied"
    summary["applied"] = True
    summary["selected_segment"] = selected_segment
    summary["selected_segments"] = selected_segments
    summary["information_capacity"] = {
        "selection_policy": "maximize_total_information_matrix_capacity",
        "selected_segment_count": int(len(selected_segments)),
        "selection_trace": selected_segment_traces,
        "final_information_matrix": running_information_matrix.tolist(),
        "final_information_metrics": running_information_metrics,
    }
    summary["selected_candidate_keys"] = _sort_candidate_key_payloads(
        selected_key_payloads
    )
    summary["selected_window_ids"] = selected_window_ids
    summary["selected_candidate_count"] = int(len(summary["selected_candidate_keys"]))
    candidate_decisions = _candidate_screening_decisions(
        candidates_by_key=candidates_by_key,
        segment_summaries=ranked_segments,
        selected_segments=selected_segments,
        selected_key_tuples=set(selected_key_tuples),
    )
    for decision in candidate_decisions:
        key_tuple = (
            int(decision["start_timestamp_ns"]),
            int(decision["end_timestamp_ns"]),
            int(decision["frame_stride"]),
        )
        if key_tuple in rescued_key_tuples:
            decision["selected_by_window_rescue"] = True
    summary["candidate_decisions"] = candidate_decisions
    return summary
