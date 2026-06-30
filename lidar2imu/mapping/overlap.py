from __future__ import annotations

import math

import numpy as np
import open3d as o3d


def point_cloud_overlap_metrics(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
) -> dict[str, float | int]:
    if source_cloud.is_empty() or target_cloud.is_empty():
        return {
            "source_point_count": int(len(source_cloud.points)),
            "target_point_count": int(len(target_cloud.points)),
            "nn_mean_m": float("nan"),
            "nn_median_m": float("nan"),
            "nn_p95_m": float("nan"),
            "within_0p1m_ratio": 0.0,
            "within_0p2m_ratio": 0.0,
            "within_0p4m_ratio": 0.0,
        }
    distances = np.asarray(source_cloud.compute_point_cloud_distance(target_cloud))
    if distances.size == 0:
        return {
            "source_point_count": int(len(source_cloud.points)),
            "target_point_count": int(len(target_cloud.points)),
            "nn_mean_m": float("nan"),
            "nn_median_m": float("nan"),
            "nn_p95_m": float("nan"),
            "within_0p1m_ratio": 0.0,
            "within_0p2m_ratio": 0.0,
            "within_0p4m_ratio": 0.0,
        }
    return {
        "source_point_count": int(len(source_cloud.points)),
        "target_point_count": int(len(target_cloud.points)),
        "nn_mean_m": float(np.mean(distances)),
        "nn_median_m": float(np.median(distances)),
        "nn_p95_m": float(np.percentile(distances, 95)),
        "within_0p1m_ratio": float(np.mean(distances <= 0.10)),
        "within_0p2m_ratio": float(np.mean(distances <= 0.20)),
        "within_0p4m_ratio": float(np.mean(distances <= 0.40)),
    }


def overlap_quality_score(metrics: dict[str, float | int]) -> float:
    within_0p4m_ratio = float(metrics.get("within_0p4m_ratio", 0.0) or 0.0)
    nn_mean_m = metrics.get("nn_mean_m")
    if nn_mean_m is None or not math.isfinite(float(nn_mean_m)):
        return 0.0
    return float(
        max(within_0p4m_ratio, 0.0) * math.exp(-max(float(nn_mean_m), 0.0) / 0.20)
    )


def passes_overlap_gate(
    metrics: dict[str, float | int],
    *,
    min_within_0p4m_ratio: float,
    max_nn_mean_m: float,
) -> bool:
    within_0p4m_ratio = float(metrics.get("within_0p4m_ratio", 0.0) or 0.0)
    nn_mean_m = metrics.get("nn_mean_m")
    if nn_mean_m is None or not math.isfinite(float(nn_mean_m)):
        return False
    return within_0p4m_ratio >= float(min_within_0p4m_ratio) and float(
        nn_mean_m
    ) <= float(max_nn_mean_m)
