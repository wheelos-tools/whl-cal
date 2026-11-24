"""Read-only provenance and correspondence snapshots for semantic L2L."""

from __future__ import annotations

from typing import Any

import numpy as np

CHANNELS = ("ground", "facade", "pole", "intensity")


def capture_features(
    trace: dict[str, Any],
    frame: Any,
    primitives: Any,
    supports: dict[str, list[np.ndarray]],
    pca: dict[str, list[np.ndarray]],
    before_cap: dict[str, int],
    voxel_counts: dict[str, int],
) -> None:
    trace["frame_points"] = np.asarray(frame.points).copy()
    for name in ("intensity", "ring", "scan_indices", "point_indices"):
        value = getattr(frame, name)
        if value is not None:
            trace[name] = np.asarray(value).copy()
    trace["stats"] = dict(primitives.stats)
    trace["before_cap"] = before_cap
    trace["voxel_counts"] = voxel_counts
    trace["features"] = {}
    for channel in CHANNELS:
        indices = supports[channel]
        offsets = np.r_[0, np.cumsum([len(i) for i in indices])].astype(np.int64)
        vectors = getattr(
            primitives,
            "pole_directions" if channel == "pole" else f"{channel}_normals",
        )
        trace["features"][channel] = {
            "points": getattr(primitives, f"{channel}_points").copy(),
            "vectors": vectors.copy(),
            "support_offsets": offsets,
            "support_indices": (
                np.concatenate(indices).astype(np.int64)
                if indices
                else np.empty(0, dtype=np.int64)
            ),
            "pca": np.asarray(pca[channel], dtype=np.float64).reshape(-1, 3),
        }


def record_candidates(
    stage: dict[str, Any] | None,
    channel: str,
    count: int,
    *,
    queried: np.ndarray | None = None,
    target: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    gates: dict[str, np.ndarray] | None = None,
    values: dict[str, np.ndarray] | None = None,
    accepted: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
    unavailable_reason: str = "no_target_features",
) -> None:
    if stage is None:
        return
    nearest = np.full(count, -1, dtype=np.int64)
    distances = np.full(count, np.nan)
    reasons = np.full(
        count,
        "outside_shared_visibility" if queried is not None else unavailable_reason,
        dtype="U40",
    )
    checks: dict[str, np.ndarray] = {}
    measurements: dict[str, np.ndarray] = {}
    if queried is not None:
        nearest[queried] = target
        distances[queried] = distance
        reasons[queried] = "accepted"
        for name, gate in (gates or {}).items():
            evaluated = np.full(count, -1, dtype=np.int8)
            evaluated[queried] = gate.astype(np.int8)
            checks[name] = evaluated
            rejected = queried[(~gate) & (reasons[queried] == "accepted")]
            reasons[rejected] = name
        for name, value in (values or {}).items():
            expanded = np.full(count, np.nan)
            expanded[queried] = value
            measurements[name] = expanded
        if accepted is not None:
            # The algorithm's selection is authoritative, including reciprocal cuts.
            selected = np.zeros(count, dtype=bool)
            selected[accepted] = True
            reasons[(reasons == "accepted") & ~selected] = "reciprocal"
    else:
        selected = np.zeros(count, dtype=bool)
    source_indices = np.flatnonzero(reasons == "accepted")
    stage["channels"][channel] = {
        "nearest_target": nearest,
        "distance_m": distances,
        "reason": reasons,
        "gates": checks,
        "values": measurements,
        "thresholds": thresholds or {},
        "source_indices": source_indices,
        "target_indices": nearest[source_indices],
    }


def geometric_residual_vectors(
    channel: str,
    difference: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    if channel in {"ground", "facade"}:
        return np.sum(difference * vectors, axis=1)[:, None]
    if channel == "pole":
        return difference - np.sum(difference * vectors, axis=1)[:, None] * vectors
    return difference


def record_stage_residuals(
    stage: dict[str, Any],
    source: Any,
    target: Any,
    transformation: np.ndarray,
    weights: dict[str, float],
    huber_scale: float,
) -> None:
    """Evaluate the frozen, actual solver pairs at the post-solve transform."""
    stage["T_after"] = transformation.copy()
    stage["optimized"] = True
    for channel, data in stage["channels"].items():
        si, ti = data["source_indices"], data["target_indices"]
        sp = getattr(source, f"{channel}_points")[si]
        tp = getattr(target, f"{channel}_points")[ti]
        diff = sp @ transformation[:3, :3].T + transformation[:3, 3] - tp
        vectors = getattr(
            target, "pole_directions" if channel == "pole" else f"{channel}_normals"
        )[ti]
        residual = geometric_residual_vectors(channel, diff, vectors)
        data["residual_vector_m"] = residual
        data["residual_norm_m"] = np.linalg.norm(residual, axis=1)
        data["euclidean_after_m"] = np.linalg.norm(diff, axis=1)
        data["channel_weight"] = weights[channel]
        if channel != "pole":
            weighted = np.abs(weights[channel] * residual)
            data["huber_component_weights"] = np.minimum(
                1.0, huber_scale / np.maximum(weighted, 1e-12)
            )


def record_pole_huber(
    stage: dict[str, Any], residual_2d: np.ndarray, weight: float, huber_scale: float
) -> None:
    data = stage["channels"]["pole"]
    data["solver_residual_components_m"] = residual_2d.reshape(-1, 2).copy()
    weighted = np.abs(weight * data["solver_residual_components_m"])
    data["huber_component_weights"] = np.minimum(
        1.0, huber_scale / np.maximum(weighted, 1e-12)
    )
