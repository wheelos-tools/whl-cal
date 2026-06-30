from __future__ import annotations

import html
import math
from typing import Any


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: Any, digits: int = 3) -> str:
    numeric = float_or_none(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def svg_line_chart(
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
            f'class="tick-label">{html.escape(format_float(y_value, y_digits))}</text>'
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
            f"{html.escape(format_float(x_value, x_digits))}</text>"
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


def svg_xy_path_chart(
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
            f'class="tick-label">{html.escape(format_float(y_value, 2))}</text>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20:.2f}" '
            f'text-anchor="middle" class="tick-label">'
            f"{html.escape(format_float(x_value, 1))}</text>"
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


def series_from_rows(rows: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    points = []
    for index, row in enumerate(rows, start=1):
        value = float_or_none(row.get(key))
        if value is None:
            continue
        points.append((float(index), value))
    return points


def cost_scan_series(cost_scan: dict[str, Any]) -> list[tuple[float, float]]:
    angles = cost_scan.get("angles_deg") or []
    costs = cost_scan.get("cost_values") or []
    if len(angles) != len(costs):
        return []
    points = []
    for angle, cost in zip(angles, costs):
        angle_value = float_or_none(angle)
        cost_value = float_or_none(cost)
        if angle_value is None or cost_value is None:
            continue
        points.append((angle_value, cost_value))
    return points


def svg_yaw_cost_chart(cost_scan: dict[str, Any]) -> str:
    points = cost_scan_series(cost_scan)
    best_cost = float_or_none(cost_scan.get("best_cost"))
    best_yaw = float_or_none(cost_scan.get("best_yaw_deg"))
    max_cost_ratio = float_or_none(cost_scan.get("max_cost_ratio"))
    plateau_width = float_or_none(cost_scan.get("within_5pct_span_deg"))
    if not points or best_cost is None or best_cost <= 0.0:
        return svg_line_chart(
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
            f'class="tick-label">{html.escape(format_float(y_value, 2))}</text>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20:.2f}" '
            f'text-anchor="middle" class="tick-label">'
            f"{html.escape(format_float(x_value, 0))}</text>"
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
        f"max ratio={format_float(max_cost_ratio, 2)}, "
        f"5% plateau={format_float(plateau_width, 1)} deg"
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
