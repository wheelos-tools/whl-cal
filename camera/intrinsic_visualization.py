#!/usr/bin/env python3

"""Rendering helpers for camera intrinsic calibration UI and review artifacts."""

import cv2
import numpy as np


def draw_text(
    image,
    text,
    position,
    color=(255, 255, 255),
    scale=1.0,
    thickness=2,
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_capture_runtime_info(image, runtime_info):
    runtime_info = runtime_info or {}
    actual = runtime_info.get("actual_capture_resolution", {})
    display = runtime_info.get("display_resolution", {})
    rendering = runtime_info.get("display_rendering", {})
    requested = runtime_info.get("requested_capture_resolution")
    lines = [
        (
            "Capture "
            f"{actual.get('width', 0)}x{actual.get('height', 0)}"
            f" | Display {display.get('width', 0)}x{display.get('height', 0)}"
        ),
        (
            "Forced capture: "
            + (
                "ON"
                if runtime_info.get("force_capture_resolution")
                else "OFF (preferred for full FOV)"
            )
        ),
    ]
    if rendering:
        lines.append(
            "Render "
            f"{rendering.get('render_width', 0)}x{rendering.get('render_height', 0)}"
            f" pad=({rendering.get('pad_x', 0)},{rendering.get('pad_y', 0)})"
        )
    if requested is not None:
        lines.append(
            "Requested capture "
            f"{requested.get('width', 0)}x{requested.get('height', 0)}"
        )
    stream_health = runtime_info.get("stream_health") or {}
    if stream_health:
        lines.append(
            "Stream reconnects="
            f"{stream_health.get('reconnect_count', 0)} invalid="
            f"{stream_health.get('total_invalid_frames', 0)}"
        )
    warnings = runtime_info.get("warnings", [])
    if warnings:
        lines.extend(warnings[:2])

    start_y = max(30, image.shape[0] - 140)
    for index, line in enumerate(lines):
        color = (0, 180, 255) if index >= 2 else (255, 255, 255)
        draw_text(
            image,
            line,
            (30, start_y + index * 30),
            color=color,
            scale=0.7,
            thickness=2,
        )


def draw_dynamic_ui(display, grid_coverage, grid_shape, feedback_text, progress):
    rows, cols = grid_shape
    height, width = display.shape[:2]
    progress = dict(progress or {})
    stage = str(progress.get("stage", "collect_coverage"))
    stage_label = {
        "collect_coverage": "Stage 1/2: cover more image regions",
        "collect_diverse_samples": "Stage 2/2: coverage done, collect diverse poses",
        "ready_to_calibrate": "Ready: sample target met",
    }.get(stage, stage)
    sample_count = int(progress.get("sample_count", 0))
    required_samples = int(progress.get("required_sample_count", 0))
    coverage_count = int(progress.get("coverage_cell_count", np.count_nonzero(grid_coverage > 0)))
    coverage_target = int(progress.get("coverage_target_cell_count", grid_coverage.size))
    guidance_summary = str(progress.get("guidance_summary") or "")
    guidance_actions = list(progress.get("guidance_actions") or [])

    panel_lines = []
    if feedback_text:
        panel_lines.append((feedback_text, (0, 255, 255), 0.9))
    panel_lines.append((stage_label, (0, 255, 255), 0.8))
    panel_lines.append(
        (
            f"Diverse samples: {sample_count}/{required_samples} | Spatial coverage: {coverage_count}/{coverage_target}",
            (0, 255, 255),
            0.8,
        )
    )
    if guidance_summary:
        panel_lines.append((guidance_summary, (0, 220, 255), 0.72))
    for line in guidance_actions[:2]:
        panel_lines.append((line, (180, 240, 255), 0.62))

    panel_x = 28
    panel_y = 24
    panel_width = min(width - 56, 920)
    panel_height = 22 + len(panel_lines) * 34
    sparsest_cells = {
        (int(cell.get("y", -1)), int(cell.get("x", -1)))
        for cell in list(progress.get("sparsest_cells") or [])
    }
    max_count = max(int(np.max(grid_coverage)) if grid_coverage.size else 0, 1)

    for row in range(rows):
        for col in range(cols):
            y0, x0 = int(row * height / rows), int(col * width / cols)
            y1, x1 = int((row + 1) * height / rows), int((col + 1) * width / cols)
            count = int(grid_coverage[row, col])
            overlay = display.copy()
            if count > 0:
                ratio = float(count) / float(max_count)
                heat_color = (
                    int(210 - 60 * ratio),
                    int(190 + 25 * ratio),
                    int(120 + 100 * ratio),
                )
                alpha = 0.15 + 0.30 * ratio
                cv2.rectangle(overlay, (x0, y0), (x1, y1), heat_color, -1)
                cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0, display)

            border_color = (0, 180, 255) if (row, col) in sparsest_cells else (70, 70, 70)
            border_thickness = 3 if (row, col) in sparsest_cells else 2
            cv2.rectangle(display, (x0, y0), (x1, y1), border_color, border_thickness)

            count_text = str(count)
            text_scale = 1.0
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                count_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_thickness,
            )
            text_x = max(x0 + 10, int((x0 + x1 - text_width) / 2))
            text_y = max(y0 + text_height + 10, int((y0 + y1 + text_height) / 2))
            text_color = (30, 30, 30) if count > 0 else (110, 110, 110)
            count_inside_panel = (
                panel_x <= text_x <= panel_x + panel_width
                and panel_y <= text_y <= panel_y + panel_height
            )
            if not count_inside_panel:
                cv2.putText(
                    display,
                    count_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    text_color,
                    text_thickness,
                )

    panel_overlay = display.copy()
    cv2.rectangle(
        panel_overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (32, 32, 32),
        -1,
    )
    cv2.addWeighted(panel_overlay, 0.55, display, 0.45, 0, display)
    cv2.rectangle(
        display,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (90, 90, 90),
        2,
    )

    for index, (line, color, scale) in enumerate(panel_lines):
        draw_text(
            display,
            line,
            (panel_x + 18, panel_y + 34 + index * 34),
            color,
            scale=scale,
            thickness=2,
        )

def draw_aprilgrid_debug(image, debug_info):
    debug_info = debug_info or {}
    if not debug_info or debug_info.get("target_type") != "aprilgrid":
        return

    marker_count = int(debug_info.get("detected_marker_count", 0))
    min_tags = int(debug_info.get("min_tags_per_frame", 0))
    matched_points = int(debug_info.get("matched_point_count", 0))
    selected_scale = debug_info.get("selected_scale")
    failure_stage = debug_info.get("failure_stage") or "ok"
    selected_ids = list(debug_info.get("selected_marker_ids") or [])
    attempts = list(debug_info.get("attempts") or [])

    lines = [
        (
            "AprilGrid "
            f"markers={marker_count}/{min_tags} points={matched_points} "
            f"scale={selected_scale if selected_scale is not None else 'n/a'} "
            f"stage={failure_stage}"
        )
    ]
    if selected_ids:
        ids_preview = ",".join(str(value) for value in selected_ids[:10])
        if len(selected_ids) > 10:
            ids_preview += ",..."
        lines.append(f"IDs: {ids_preview}")
    if attempts:
        attempt_summary = " ".join(
            f"{float(item.get('scale', 0.0)):.2f}x:{int(item.get('detected_marker_count', 0))}"
            for item in attempts[:6]
        )
        lines.append(f"Attempts {attempt_summary}")

    start_y = 210
    for index, line in enumerate(lines):
        draw_text(
            image,
            line,
            (50, start_y + index * 35),
            color=(255, 220, 120),
            scale=0.75,
            thickness=2,
        )


def draw_valid_roi(image, preview_info):
    roi = (preview_info or {}).get("valid_roi") or {}
    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    roi_w = int(roi.get("width", 0))
    roi_h = int(roi.get("height", 0))
    annotated = image.copy()
    if roi_w > 0 and roi_h > 0:
        cv2.rectangle(
            annotated,
            (x, y),
            (x + roi_w - 1, y + roi_h - 1),
            (0, 255, 0),
            2,
        )
    return annotated


def generate_grid_overlay(image_size_wh, grid_shape):
    width, height = int(image_size_wh[0]), int(image_size_wh[1])
    rows, cols = grid_shape
    overlay = np.zeros((height, width, 3), np.uint8)
    color = (255, 100, 100)
    thickness = 2
    for row in range(rows + 1):
        y = int(round(row * height / rows))
        cv2.line(overlay, (0, y), (width, y), color, thickness)
    for col in range(cols + 1):
        x = int(round(col * width / cols))
        cv2.line(overlay, (x, 0), (x, height), color, thickness)
    return overlay


def render_preserving_aspect_ratio(display, window_size_wh):
    window_width, window_height = map(int, window_size_wh)
    src_h, src_w = display.shape[:2]
    if src_w == 0 or src_h == 0:
        return display
    scale = min(window_width / src_w, window_height / src_h)
    render_width = max(1, int(src_w * scale))
    render_height = max(1, int(src_h * scale))
    resized = cv2.resize(display, (render_width, render_height))
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    x0 = (window_width - render_width) // 2
    y0 = (window_height - render_height) // 2
    canvas[y0 : y0 + render_height, x0 : x0 + render_width] = resized
    return canvas


def build_comparison_canvas(distorted_image, undistorted_image, preview_info):
    height, width = distorted_image.shape[:2]
    canvas = np.full((height, width * 2 + 60, 3), 40, np.uint8)
    canvas[:, 20 : 20 + width] = distorted_image
    canvas[:, 40 + width : 40 + 2 * width] = undistorted_image
    draw_text(canvas, "Distorted", (50, 50), (200, 200, 255))
    draw_text(
        canvas,
        f"Undistorted alpha={preview_info['alpha']:.2f}",
        (width + 80, 50),
        (180, 255, 180),
    )
    draw_text(
        canvas,
        "Green ROI shows the all-valid crop window",
        (width + 80, 95),
        (180, 255, 180),
    )
    return canvas
