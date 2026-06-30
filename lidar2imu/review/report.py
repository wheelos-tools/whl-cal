from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

from lidar2imu.review.charts import format_float


def write_review_html(
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
        ("IMU vs LiDAR BEV trajectory", "trajectory_overlay_plot"),
        ("IMU vs LiDAR BEV position gap", "trajectory_position_gap_plot"),
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
        ("Registration review YAML", "registration_review_yaml"),
        ("Registration review CSV", "registration_review_csv"),
        ("IMU-predicted scene cloud", "imu_trajectory_cloud"),
        ("LiDAR-registered scene cloud", "lidar_trajectory_cloud"),
        ("Overlay scene cloud", "trajectory_overlay_cloud"),
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
            "Open the scene-cloud PLY files in CloudCompare or Open3D. Gray is the "
            "target scene, blue is the calibrated IMU-predicted source scene, and "
            "red is the LiDAR-registered source scene."
            "</p>"
        )

    final_yaw = html.escape(
        format_float((summary.get("final_euler_deg") or {}).get("yaw"), 3)
    )
    motion_recommendation = html.escape(
        str(motion_assessment.get("recommendation") or "-")
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
  <p class="muted">This page is intentionally scene-first: inspect the SLAM-like local clouds and the IMU-vs-LiDAR BEV trajectory before reading any other diagnostics.</p>
  <div class="card-grid">
    <div class="card">
      <strong>motion recommendation</strong><br />{motion_recommendation}
    </div>
    <div class="card"><strong>final yaw</strong><br />{final_yaw} deg</div>
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
