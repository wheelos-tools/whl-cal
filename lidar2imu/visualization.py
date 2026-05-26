#!/usr/bin/env python3

from __future__ import annotations

import html
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from lidar2imu.models import CalibrationDataset
from lidar2lidar.record_utils import PointCloudMeta, load_pointcloud_from_meta

_SVG_STYLE_BLOCK = """
<style>
.chart-title { font: 700 18px Arial, sans-serif; fill: #111827; }
.chart-note { font: 12px Arial, sans-serif; fill: #4b5563; }
.axis { stroke: #6b7280; stroke-width: 1.25; }
.grid { stroke: #d1d5db; stroke-width: 1; stroke-dasharray: 3 4; }
.reference-line { stroke-width: 1.5; stroke-dasharray: 6 4; }
.axis-label { font: 13px Arial, sans-serif; fill: #374151; }
.tick-label { font: 12px Arial, sans-serif; fill: #374151; }
.value-label { font: 12px Arial, sans-serif; fill: #111827; }
.marker-outline { stroke: #ffffff; stroke-width: 1.5; }
</style>
""".strip()


@dataclass(frozen=True)
class _TrajectoryNode:
    timestamp_ns: int
    record_path: str | None
    imu_pose: np.ndarray
    lidar_pose: np.ndarray


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, digits: int = 3) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _svg_line_chart(
    title: str,
    *,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float]],
    width: int = 860,
    height: int = 260,
    color: str = "#2563eb",
    y_min: float | None = None,
    y_max: float | None = None,
    x_digits: int = 1,
    y_digits: int = 3,
    note_text: str | None = None,
) -> str:
    escaped_title = html.escape(title)
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="{escaped_title}">'
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            f'class="chart-title">{escaped_title}</text>'
            '<text x="24" y="52" class="muted">No data.</text>'
            "</svg>"
        )

    margin_left = 72
    margin_right = 24
    margin_top = 42
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys) if y_min is None else float(y_min)
    max_y = max(ys) if y_max is None else float(y_max)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        padding = 1.0 if math.isclose(min_y, 0.0) else abs(min_y) * 0.1
        min_y -= padding
        max_y += padding

    def _scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def _scale_y(value: float) -> float:
        return (
            margin_top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        )

    polyline = " ".join(
        f"{_scale_x(x_value):.2f},{_scale_y(y_value):.2f}"
        for x_value, y_value in points
    )
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="{escaped_title}">',
        (
            f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" '
            f'class="chart-title">{escaped_title}</text>'
        ),
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{width - margin_right}" y2="{margin_top + plot_height}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" class="axis" />',
    ]
    tick_count = 5
    for tick_index in range(tick_count):
        fraction = tick_index / max(tick_count - 1, 1)
        y_value = min_y + (max_y - min_y) * fraction
        y_pos = _scale_y(y_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" '
            f'x2="{width - margin_right}" y2="{y_pos:.2f}" class="grid" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
            f'class="tick-label">{html.escape(_format_float(y_value, y_digits))}</text>'
        )
        x_value = min_x + (max_x - min_x) * fraction
        x_pos = _scale_x(x_value)
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" '
            f'x2="{x_pos:.2f}" y2="{margin_top + plot_height}" class="grid" />'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20:.2f}" '
            f'text-anchor="middle" class="tick-label">'
            f"{html.escape(_format_float(x_value, x_digits))}</text>"
        )
    parts.append(
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
    )
    for x_value, y_value in points:
        parts.append(
            f'<circle cx="{_scale_x(x_value):.2f}" cy="{_scale_y(y_value):.2f}" '
            f'r="3.5" fill="{color}" opacity="0.95" />'
        )
    parts.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" '
            f'class="axis-label">{html.escape(x_label)}</text>',
            f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18,{height / 2:.1f})" class="axis-label">'
            f"{html.escape(y_label)}</text>",
        ]
    )
    if note_text:
        parts.append(
            f'<text x="{width - margin_right}" y="22" text-anchor="end" '
            f'class="chart-note">{html.escape(note_text)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_xy_path_chart(
    title: str,
    *,
    series: list[dict[str, Any]],
    x_label: str = "x (m)",
    y_label: str = "y (m)",
    width: int = 860,
    height: int = 420,
) -> str:
    escaped_title = html.escape(title)
    valid_series = [item for item in series if item.get("points")]
    if not valid_series:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="{escaped_title}">'
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            f'class="chart-title">{escaped_title}</text>'
            '<text x="24" y="52" class="muted">No trajectory data.</text>'
            "</svg>"
        )

    margin_left = 72
    margin_right = 24
    margin_top = 42
    margin_bottom = 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    xs = [point[0] for item in valid_series for point in item["points"]]
    ys = [point[1] for item in valid_series for point in item["points"]]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = max(span_x / plot_width, span_y / plot_height)
    padded_span_x = plot_width * scale
    padded_span_y = plot_height * scale
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    min_x = center_x - padded_span_x * 0.5
    max_x = center_x + padded_span_x * 0.5
    min_y = center_y - padded_span_y * 0.5
    max_y = center_y + padded_span_y * 0.5

    def _scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def _scale_y(value: float) -> float:
        return (
            margin_top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="{escaped_title}">',
        (
            f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" '
            f'class="chart-title">{escaped_title}</text>'
        ),
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{width - margin_right}" y2="{margin_top + plot_height}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" class="axis" />',
    ]
    tick_count = 5
    for tick_index in range(tick_count):
        fraction = tick_index / max(tick_count - 1, 1)
        y_value = min_y + (max_y - min_y) * fraction
        x_value = min_x + (max_x - min_x) * fraction
        y_pos = _scale_y(y_value)
        x_pos = _scale_x(x_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" '
            f'x2="{width - margin_right}" y2="{y_pos:.2f}" class="grid" />'
        )
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" '
            f'x2="{x_pos:.2f}" y2="{margin_top + plot_height}" class="grid" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
            f'class="tick-label">{html.escape(_format_float(y_value, 2))}</text>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20:.2f}" '
            f'text-anchor="middle" class="tick-label">'
            f"{html.escape(_format_float(x_value, 1))}</text>"
        )
    for item in valid_series:
        name = str(item["name"])
        points = item["points"]
        color = str(item["color"])
        stroke_width = float(item.get("stroke_width", 2.5))
        opacity = float(item.get("opacity", 0.95))
        dasharray = item.get("dasharray")
        polyline = " ".join(
            f"{_scale_x(x_value):.2f},{_scale_y(y_value):.2f}"
            for x_value, y_value in points
        )
        dash_attr = "" if dasharray is None else f' stroke-dasharray="{dasharray}"'
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width:.1f}" stroke-opacity="{opacity:.2f}"'
            f'{dash_attr} stroke-linejoin="round" stroke-linecap="round" />'
        )
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        parts.append(
            f'<circle cx="{_scale_x(start_x):.2f}" cy="{_scale_y(start_y):.2f}" '
            f'r="4.5" fill="{color}" opacity="{opacity:.2f}" class="marker-outline" />'
        )
        parts.append(
            f'<circle cx="{_scale_x(end_x):.2f}" cy="{_scale_y(end_y):.2f}" '
            f'r="6.0" fill="{color}" opacity="{opacity:.2f}" class="marker-outline" />'
        )
        parts.append(
            f'<text x="{_scale_x(end_x) + 8:.2f}" y="{_scale_y(end_y) - 8:.2f}" '
            f'class="tick-label">{html.escape(name)}</text>'
        )

    legend_x = width - margin_right - 180
    legend_y = margin_top + 6
    for index, item in enumerate(valid_series):
        name = str(item["name"])
        color = str(item["color"])
        dasharray = item.get("dasharray")
        row_y = legend_y + index * 18
        parts.append(
            f'<line x1="{legend_x}" y1="{row_y - 3}" x2="{legend_x + 14}" '
            f'y2="{row_y - 3}" stroke="{color}" stroke-width="3" '
            + ("" if dasharray is None else f'stroke-dasharray="{dasharray}" ')
            + "/>"
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="{row_y + 1}" class="tick-label">'
            f"{html.escape(name)}</text>"
        )

    parts.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" '
            f'class="axis-label">{html.escape(x_label)}</text>',
            f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18,{height / 2:.1f})" class="axis-label">'
            f"{html.escape(y_label)}</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def _write_svg(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload:
        payload = payload.replace(
            "<svg ", "<svg xmlns='http://www.w3.org/2000/svg' ", 1
        )
        payload = payload.replace(">", f">{_SVG_STYLE_BLOCK}", 1)
    path.write_text(
        ("<svg xmlns='http://www.w3.org/2000/svg'></svg>" if not payload else payload),
        encoding="utf-8",
    )
    return str(path)


def _series_from_rows(
    rows: list[dict[str, Any]], key: str
) -> list[tuple[float, float]]:
    points = []
    for index, row in enumerate(rows, start=1):
        value = _float_or_none(row.get(key))
        if value is None:
            continue
        points.append((float(index), value))
    return points


def _cost_scan_series(cost_scan: dict[str, Any]) -> list[tuple[float, float]]:
    angles = cost_scan.get("angles_deg") or []
    costs = cost_scan.get("cost_values") or []
    if len(angles) != len(costs):
        return []
    points = []
    for angle, cost in zip(angles, costs):
        angle_value = _float_or_none(angle)
        cost_value = _float_or_none(cost)
        if angle_value is None or cost_value is None:
            continue
        points.append((angle_value, cost_value))
    return points


def _svg_yaw_cost_chart(cost_scan: dict[str, Any]) -> str:
    points = _cost_scan_series(cost_scan)
    best_cost = _float_or_none(cost_scan.get("best_cost"))
    best_yaw = _float_or_none(cost_scan.get("best_yaw_deg"))
    max_cost_ratio = _float_or_none(cost_scan.get("max_cost_ratio"))
    plateau_width = _float_or_none(cost_scan.get("within_5pct_span_deg"))
    if not points or best_cost is None or best_cost <= 0.0:
        return _svg_line_chart(
            "Yaw cost ratio scan",
            x_label="yaw (deg)",
            y_label="cost / best cost",
            points=[],
        )

    ratio_points = [(angle, cost / best_cost) for angle, cost in points]
    width = 860
    height = 300
    margin_left = 72
    margin_right = 24
    margin_top = 42
    margin_bottom = 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_x = min(point[0] for point in ratio_points)
    max_x = max(point[0] for point in ratio_points)
    min_y = 1.0
    max_y = max(max(point[1] for point in ratio_points), 1.2)
    max_y *= 1.05

    def _scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * plot_width

    def _scale_y(value: float) -> float:
        return (
            margin_top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        )

    ratio_polyline = " ".join(
        f"{_scale_x(angle):.2f},{_scale_y(value):.2f}" for angle, value in ratio_points
    )
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        'aria-label="Yaw cost ratio scan">',
        '<text x="430" y="22" text-anchor="middle" class="chart-title">'
        "Yaw cost ratio scan"
        "</text>",
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{width - margin_right}" y2="{margin_top + plot_height}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" class="axis" />',
    ]
    tick_count = 5
    for tick_index in range(tick_count):
        fraction = tick_index / max(tick_count - 1, 1)
        y_value = min_y + (max_y - min_y) * fraction
        y_pos = _scale_y(y_value)
        x_value = min_x + (max_x - min_x) * fraction
        x_pos = _scale_x(x_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" '
            f'x2="{width - margin_right}" y2="{y_pos:.2f}" class="grid" />'
        )
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" '
            f'x2="{x_pos:.2f}" y2="{margin_top + plot_height}" class="grid" />'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
            f'class="tick-label">{html.escape(_format_float(y_value, 2))}</text>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20:.2f}" '
            f'text-anchor="middle" class="tick-label">'
            f"{html.escape(_format_float(x_value, 0))}</text>"
        )

    threshold_y = 1.05
    parts.append(
        f'<line x1="{margin_left}" y1="{_scale_y(threshold_y):.2f}" '
        f'x2="{width - margin_right}" y2="{_scale_y(threshold_y):.2f}" '
        'class="reference-line" stroke="#16a34a" />'
    )
    parts.append(
        f'<text x="{width - margin_right - 8}" y="{_scale_y(threshold_y) - 6:.2f}" '
        'text-anchor="end" class="tick-label">+5% threshold</text>'
    )
    if best_yaw is not None:
        parts.append(
            f'<line x1="{_scale_x(best_yaw):.2f}" y1="{margin_top}" '
            f'x2="{_scale_x(best_yaw):.2f}" y2="{margin_top + plot_height}" '
            'class="reference-line" stroke="#6b7280" />'
        )
        parts.append(
            f'<text x="{_scale_x(best_yaw):.2f}" y="{margin_top + 14:.2f}" '
            'text-anchor="middle" class="tick-label">best yaw</text>'
        )

    parts.append(
        f'<polyline points="{ratio_polyline}" fill="none" stroke="#0f766e" '
        'stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round" />'
    )
    for angle, value in ratio_points[:: max(len(ratio_points) // 40, 1)]:
        parts.append(
            f'<circle cx="{_scale_x(angle):.2f}" cy="{_scale_y(value):.2f}" '
            'r="2.2" fill="#0f766e" />'
        )

    note_text = (
        f"max ratio={_format_float(max_cost_ratio, 2)}, "
        f"5% plateau={_format_float(plateau_width, 1)} deg"
    )
    parts.extend(
        [
            '<text x="430" y="292" text-anchor="middle" class="axis-label">'
            "yaw (deg)"
            "</text>",
            '<text x="18" y="150" text-anchor="middle" '
            'transform="rotate(-90 18,150)" class="axis-label">'
            "cost / best cost"
            "</text>",
            f'<text x="{width - margin_right}" y="22" text-anchor="end" '
            f'class="chart-note">{html.escape(note_text)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _matrix_from_components(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _trajectory_nodes(
    dataset: CalibrationDataset, final_transform: np.ndarray
) -> list[_TrajectoryNode]:
    ordered_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not ordered_samples:
        return []

    final_transform = np.asarray(final_transform, dtype=float)
    final_transform_inv = np.linalg.inv(final_transform)
    imu_pose = np.eye(4, dtype=float)
    lidar_pose = np.eye(4, dtype=float)
    nodes = [
        _TrajectoryNode(
            timestamp_ns=int(ordered_samples[0].start_timestamp_ns),
            record_path=ordered_samples[0].metadata.get("record_path_start"),
            imu_pose=imu_pose.copy(),
            lidar_pose=lidar_pose.copy(),
        )
    ]
    for sample in ordered_samples:
        imu_delta = _matrix_from_components(
            sample.imu_delta_rotation, sample.imu_delta_translation
        )
        lidar_delta = _matrix_from_components(
            sample.lidar_delta_rotation, sample.lidar_delta_translation
        )
        lidar_delta_in_imu = final_transform @ lidar_delta @ final_transform_inv
        imu_pose = imu_pose @ imu_delta
        lidar_pose = lidar_pose @ lidar_delta_in_imu
        nodes.append(
            _TrajectoryNode(
                timestamp_ns=int(sample.end_timestamp_ns),
                record_path=sample.metadata.get("record_path_end"),
                imu_pose=imu_pose.copy(),
                lidar_pose=lidar_pose.copy(),
            )
        )
    return nodes


def _trajectory_xy_points(
    nodes: list[_TrajectoryNode], *, source: str
) -> list[tuple[float, float]]:
    points = []
    for node in nodes:
        pose = node.imu_pose if source == "imu" else node.lidar_pose
        points.append((float(pose[0, 3]), float(pose[1, 3])))
    return points


def _trajectory_gap_points(nodes: list[_TrajectoryNode]) -> list[tuple[float, float]]:
    points = []
    for index, node in enumerate(nodes):
        gap = float(np.linalg.norm(node.imu_pose[:3, 3] - node.lidar_pose[:3, 3]))
        points.append((float(index), gap))
    return points


def _write_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write point cloud: {path}")
    return str(path)


def _load_node_clouds(
    dataset: CalibrationDataset, nodes: list[_TrajectoryNode]
) -> list[tuple[_TrajectoryNode, o3d.geometry.PointCloud]]:
    if not nodes:
        return []
    motion_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not motion_samples:
        return []
    lidar_topic = motion_samples[0].metadata.get("lidar_topic")
    if not lidar_topic:
        return []

    node_clouds = []
    for node in nodes:
        if not node.record_path:
            return []
        if not Path(node.record_path).exists():
            return []
        cloud = load_pointcloud_from_meta(
            PointCloudMeta(
                topic=str(lidar_topic),
                frame_id=dataset.child_frame,
                timestamp_ns=int(node.timestamp_ns),
                record_path=str(node.record_path),
            )
        )
        if cloud.is_empty():
            continue
        cloud = cloud.voxel_down_sample(0.20)
        node_clouds.append((node, cloud))
    return node_clouds


def _build_stitched_cloud(
    node_clouds: list[tuple[_TrajectoryNode, o3d.geometry.PointCloud]],
    *,
    pose_source: str,
    final_transform: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for node, cloud in node_clouds:
        transformed = o3d.geometry.PointCloud(cloud)
        base_pose = node.imu_pose if pose_source == "imu" else node.lidar_pose
        transformed.transform(base_pose @ final_transform)
        transformed.paint_uniform_color(list(color))
        merged += transformed
    if merged.is_empty():
        return merged
    return merged.voxel_down_sample(0.20)


def _write_review_html(
    path: Path,
    *,
    summary: dict[str, Any],
    final_acceptance: dict[str, Any],
    motion_assessment: dict[str, Any],
    artifacts: dict[str, str],
) -> str:
    def _rel_link(artifact: str) -> str:
        return os.path.relpath(str(Path(artifact)), start=str(path.parent))

    sections = []
    ordered_images = [
        ("Ground normal residuals", "ground_residuals_plot"),
        ("Ground height residuals", "ground_height_residuals_plot"),
        ("Motion rotation residuals", "motion_rotation_residuals_plot"),
        ("Motion translation residuals", "motion_residuals_plot"),
        ("Motion registration fitness", "motion_registration_fitness_plot"),
        ("IMU vs LiDAR relative trajectory", "trajectory_overlay_plot"),
        ("IMU vs LiDAR cumulative position gap", "trajectory_position_gap_plot"),
        ("Holdout motion residuals", "holdout_motion_residuals_plot"),
        ("Yaw cost scan", "yaw_cost_scan"),
    ]
    for title, key in ordered_images:
        artifact = artifacts.get(key)
        if artifact is None:
            continue
        sections.append(
            (
                "<section>"
                f"<h2>{html.escape(title)}</h2>"
                f'<img src="{html.escape(_rel_link(artifact))}" '
                f'alt="{html.escape(title)}" />'
                "</section>"
            )
        )

    linked_files = []
    for label, key in (
        ("Acceptance report", "acceptance_report"),
        ("Metrics", "metrics"),
        ("Data quality", "data_quality"),
        ("Observability", "observability"),
        ("Ground residuals CSV", "ground_residuals_csv"),
        ("Motion residuals CSV", "motion_residuals_csv"),
        ("Holdout motion residuals CSV", "holdout_motion_residuals_csv"),
        ("IMU stitched cloud", "imu_trajectory_cloud"),
        ("LiDAR stitched cloud", "lidar_trajectory_cloud"),
        ("Overlay stitched cloud", "trajectory_overlay_cloud"),
    ):
        artifact = artifacts.get(key)
        if artifact is None:
            continue
        linked_files.append(
            (
                f'<li><a href="{html.escape(_rel_link(artifact))}">'
                f"{html.escape(label)}</a></li>"
            )
        )

    cloud_hint = ""
    if artifacts.get("trajectory_overlay_cloud") is not None:
        cloud_hint = (
            '<p class="muted">'
            "Open the stitched-cloud PLY files in CloudCompare or Open3D to inspect "
            "how the selected motion snippets align under IMU and LiDAR odometry."
            "</p>"
        )

    final_acceptance_status = html.escape(
        str(summary.get("final_acceptance_status") or "-")
    )
    release_ready = html.escape(str(summary.get("release_ready") or False))
    final_recommendation = html.escape(
        str(final_acceptance.get("recommendation") or "-")
    )
    motion_recommendation = html.escape(
        str(motion_assessment.get("recommendation") or "-")
    )
    final_yaw = html.escape(
        _format_float((summary.get("final_euler_deg") or {}).get("yaw"), 3)
    )
    delta_to_reference = summary.get("delta_to_reference") or {}
    delta_to_reference_translation = html.escape(
        _format_float(delta_to_reference.get("translation_norm_m"), 3)
    )
    delta_to_reference_rotation = html.escape(
        _format_float(delta_to_reference.get("rotation_deg"), 3)
    )
    delta_to_reference_text = (
        f"{delta_to_reference_translation} m / " f"{delta_to_reference_rotation} deg"
    )

    html_payload = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>lidar2imu review report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px 14px;
      background: #fafafa;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
    }}
    .pass {{ background: #dcfce7; color: #166534; }}
    .warning {{ background: #fef3c7; color: #92400e; }}
    .fail {{ background: #fee2e2; color: #991b1b; }}
    img {{
      width: 100%;
      max-width: 900px;
      border: 1px solid #e5e7eb;
      background: #fff;
      margin-bottom: 20px;
    }}
    ul {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>lidar2imu visual review</h1>
  <p class="muted">Open this file in a browser to review the run visually.</p>
  <div class="card-grid">
    <div class="card"><strong>acceptance</strong><br />{final_acceptance_status}</div>
    <div class="card"><strong>release ready</strong><br />{release_ready}</div>
    <div class="card">
      <strong>final recommendation</strong><br />{final_recommendation}
    </div>
    <div class="card">
      <strong>motion recommendation</strong><br />{motion_recommendation}
    </div>
    <div class="card"><strong>final yaw</strong><br />{final_yaw} deg</div>
    <div class="card">
      <strong>delta to reference</strong><br />{delta_to_reference_text}
    </div>
  </div>
  <section>
    <h2>Files to inspect</h2>
    <ul>{''.join(linked_files)}</ul>
    {cloud_hint}
  </section>
  {''.join(sections)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_payload, encoding="utf-8")
    return str(path)


def build_visualization_artifacts(
    diagnostics_dir: Path,
    *,
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    summary: dict[str, Any],
    final_acceptance: dict[str, Any],
    motion_assessment: dict[str, Any],
    ground_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]],
    holdout_motion_rows: list[dict[str, Any]],
    observability: dict[str, Any],
    artifact_links: dict[str, str],
) -> dict[str, str]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    ground_plot = _write_svg(
        diagnostics_dir / "ground_residuals_plot.svg",
        _svg_line_chart(
            "Ground residual angle by sample",
            x_label="sample index",
            y_label="normal angle (deg)",
            points=_series_from_rows(ground_rows, "normal_angle_deg"),
            color="#2563eb",
        ),
    )
    artifacts["ground_residuals_plot"] = ground_plot
    artifacts["ground_height_residuals_plot"] = _write_svg(
        diagnostics_dir / "ground_height_residuals_plot.svg",
        _svg_line_chart(
            "Ground height residual by sample",
            x_label="sample index",
            y_label="height residual (m)",
            points=_series_from_rows(ground_rows, "height_residual_m"),
            color="#ea580c",
        ),
    )
    artifacts["motion_rotation_residuals_plot"] = _write_svg(
        diagnostics_dir / "motion_rotation_residuals_plot.svg",
        _svg_line_chart(
            "Motion rotation residual by sample",
            x_label="sample index",
            y_label="rotation residual (deg)",
            points=_series_from_rows(motion_rows, "rotation_residual_deg"),
            color="#2563eb",
        ),
    )
    motion_plot = _write_svg(
        diagnostics_dir / "motion_residuals_plot.svg",
        _svg_line_chart(
            "Motion translation residual by sample",
            x_label="sample index",
            y_label="translation residual (m)",
            points=_series_from_rows(motion_rows, "translation_residual_m"),
            color="#dc2626",
        ),
    )
    artifacts["motion_residuals_plot"] = motion_plot
    artifacts["motion_registration_fitness_plot"] = _write_svg(
        diagnostics_dir / "motion_registration_fitness_plot.svg",
        _svg_line_chart(
            "Motion registration fitness by sample",
            x_label="sample index",
            y_label="registration fitness",
            points=_series_from_rows(motion_rows, "registration_fitness"),
            color="#16a34a",
        ),
    )

    trajectory_nodes = _trajectory_nodes(
        dataset, np.asarray(final_transform, dtype=float)
    )
    if trajectory_nodes:
        artifacts["trajectory_overlay_plot"] = _write_svg(
            diagnostics_dir / "trajectory_overlay.svg",
            _svg_xy_path_chart(
                "IMU vs LiDAR relative trajectory",
                series=[
                    {
                        "name": "LiDAR odometry",
                        "points": _trajectory_xy_points(
                            trajectory_nodes, source="lidar"
                        ),
                        "color": "#dc2626",
                        "stroke_width": 2.5,
                        "opacity": 0.95,
                    },
                    {
                        "name": "IMU odometry",
                        "points": _trajectory_xy_points(trajectory_nodes, source="imu"),
                        "color": "#2563eb",
                        "stroke_width": 5.0,
                        "dasharray": "10 7",
                        "opacity": 0.72,
                    },
                ],
            ),
        )
        artifacts["trajectory_position_gap_plot"] = _write_svg(
            diagnostics_dir / "trajectory_position_gap_plot.svg",
            _svg_line_chart(
                "Cumulative IMU vs LiDAR position gap",
                x_label="trajectory node index",
                y_label="position gap (m)",
                points=_trajectory_gap_points(trajectory_nodes),
                color="#7c3aed",
                x_digits=0,
                note_text=(
                    "single purple curve = cumulative relative-position gap between "
                    "the IMU chain and LiDAR chain"
                ),
            ),
        )

        node_clouds = _load_node_clouds(dataset, trajectory_nodes)
        if node_clouds:
            imu_cloud = _build_stitched_cloud(
                node_clouds,
                pose_source="imu",
                final_transform=np.asarray(final_transform, dtype=float),
                color=(0.145, 0.388, 0.922),
            )
            lidar_cloud = _build_stitched_cloud(
                node_clouds,
                pose_source="lidar",
                final_transform=np.asarray(final_transform, dtype=float),
                color=(0.862, 0.239, 0.176),
            )
            if not imu_cloud.is_empty():
                artifacts["imu_trajectory_cloud"] = _write_point_cloud(
                    diagnostics_dir / "imu_trajectory_cloud.ply",
                    imu_cloud,
                )
            if not lidar_cloud.is_empty():
                artifacts["lidar_trajectory_cloud"] = _write_point_cloud(
                    diagnostics_dir / "lidar_trajectory_cloud.ply",
                    lidar_cloud,
                )
            if not imu_cloud.is_empty() and not lidar_cloud.is_empty():
                overlay_cloud = imu_cloud + lidar_cloud
                artifacts["trajectory_overlay_cloud"] = _write_point_cloud(
                    diagnostics_dir / "trajectory_overlay_cloud.ply",
                    overlay_cloud.voxel_down_sample(0.20),
                )

    holdout_plot = _write_svg(
        diagnostics_dir / "holdout_motion_residuals_plot.svg",
        _svg_line_chart(
            "Holdout motion translation residual by sample",
            x_label="sample index",
            y_label="translation residual (m)",
            points=_series_from_rows(holdout_motion_rows, "translation_residual_m"),
            color="#7c3aed",
        ),
    )
    if holdout_motion_rows:
        artifacts["holdout_motion_residuals_plot"] = holdout_plot
    cost_scan = (observability.get("motion_rotation") or {}).get("cost_scan") or {}
    yaw_points = _cost_scan_series(cost_scan)
    if yaw_points:
        artifacts["yaw_cost_scan"] = _write_svg(
            diagnostics_dir / "yaw_cost_scan.svg",
            _svg_yaw_cost_chart(cost_scan),
        )
    artifacts["review_report"] = _write_review_html(
        diagnostics_dir / "review_report.html",
        summary=summary,
        final_acceptance=final_acceptance,
        motion_assessment=motion_assessment,
        artifacts={**artifact_links, **artifacts},
    )
    return artifacts
