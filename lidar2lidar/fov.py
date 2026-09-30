"""Angular support helpers for field-of-view registration experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngularSupport:
    start_deg: float
    span_deg: float
    coverage_ratio: float
    point_count: int
    margin_deg: float

    def contains(self, angles_deg: np.ndarray) -> np.ndarray:
        angles = np.mod(np.asarray(angles_deg, dtype=float), 360.0)
        if self.span_deg >= 360.0:
            return np.isfinite(angles)
        offset = np.mod(angles - self.start_deg, 360.0)
        return np.isfinite(angles) & (offset <= self.span_deg)

    def as_dict(self) -> dict:
        return {
            "start_deg": float(self.start_deg),
            "span_deg": float(self.span_deg),
            "coverage_ratio": float(self.coverage_ratio),
            "point_count": int(self.point_count),
            "margin_deg": float(self.margin_deg),
        }


def estimate_angular_support(
    point_sets: list[np.ndarray],
    *,
    coverage_ratio: float = 0.995,
    margin_deg: float = 1.0,
) -> AngularSupport:
    """Estimate a circular azimuth interval containing the requested point ratio.

    This describes observed angular support in the supplied captures. It is not
    a substitute for a manufacturer's rated field of view.
    """
    if not 0.0 < float(coverage_ratio) <= 1.0:
        raise ValueError("coverage_ratio must be in (0, 1].")
    if float(margin_deg) < 0.0:
        raise ValueError("margin_deg must be non-negative.")

    valid_sets = []
    for points in point_sets:
        xyz = np.asarray(points, dtype=float).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        planar_range = np.hypot(xyz[:, 0], xyz[:, 1])
        usable = planar_range > 1e-8
        if np.any(usable):
            valid_sets.append(xyz[usable])
    if not valid_sets:
        raise ValueError("Cannot estimate angular support without finite points.")

    points = np.concatenate(valid_sets, axis=0)
    angles = np.mod(
        np.degrees(np.arctan2(points[:, 1], points[:, 0])),
        360.0,
    )
    angles.sort()

    count = int(angles.size)
    retained_count = min(count, max(1, int(np.ceil(count * coverage_ratio))))
    extended = np.concatenate((angles, angles + 360.0))
    starts = np.arange(count)
    spans = extended[starts + retained_count - 1] - extended[starts]
    start_index = int(np.argmin(spans))
    observed_span = float(spans[start_index])
    padded_span = min(360.0, observed_span + 2.0 * float(margin_deg))
    start_deg = float((extended[start_index] - float(margin_deg)) % 360.0)
    if padded_span >= 360.0:
        start_deg = 0.0

    return AngularSupport(
        start_deg=start_deg,
        span_deg=float(padded_span),
        coverage_ratio=float(retained_count / count),
        point_count=count,
        margin_deg=float(margin_deg),
    )


def shared_fov_clouds(
    source_cloud,
    target_cloud,
    source_to_target: np.ndarray,
    source_support: AngularSupport,
    target_support: AngularSupport,
):
    """Keep each cloud's points that lie inside the other sensor's azimuth support."""
    transform = np.asarray(source_to_target, dtype=float).reshape(4, 4)
    source_points = np.asarray(source_cloud.points, dtype=float)
    target_points = np.asarray(target_cloud.points, dtype=float)
    if not np.isfinite(transform).all():
        raise ValueError("source_to_target must contain only finite values.")

    source_in_target = source_points @ transform[:3, :3].T + transform[:3, 3]
    target_to_source = np.linalg.inv(transform)
    target_in_source = (
        target_points @ target_to_source[:3, :3].T + target_to_source[:3, 3]
    )
    source_angles_in_target = np.degrees(
        np.arctan2(source_in_target[:, 1], source_in_target[:, 0])
    )
    target_angles_in_source = np.degrees(
        np.arctan2(target_in_source[:, 1], target_in_source[:, 0])
    )
    source_indices = np.flatnonzero(target_support.contains(source_angles_in_target))
    target_indices = np.flatnonzero(source_support.contains(target_angles_in_source))
    return (
        source_cloud.select_by_index(source_indices.tolist()),
        target_cloud.select_by_index(target_indices.tolist()),
        {
            "source_points_before": int(len(source_points)),
            "source_points_after": int(len(source_indices)),
            "source_retained_ratio": float(
                len(source_indices) / max(len(source_points), 1)
            ),
            "target_points_before": int(len(target_points)),
            "target_points_after": int(len(target_indices)),
            "target_retained_ratio": float(
                len(target_indices) / max(len(target_points), 1)
            ),
        },
    )
