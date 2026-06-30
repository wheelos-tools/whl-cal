from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import Any

import open3d as o3d

SVG_STYLE_BLOCK = """
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


def write_svg(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload:
        payload = payload.replace(
            "<svg ", "<svg xmlns='http://www.w3.org/2000/svg' ", 1
        )
        payload = payload.replace(">", f">{SVG_STYLE_BLOCK}", 1)
    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>" if not payload else payload,
        encoding="utf-8",
    )
    return str(path)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def write_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write point cloud: {path}")
    return str(path)
