#!/usr/bin/env python3

"""Sampling state and coverage tracking for camera intrinsic calibration."""

import time

import numpy as np


class IntrinsicSamplingState:
    """Owns accepted samples, coverage, and capture stability state."""

    def __init__(self, auto_capture_cfg):
        self.grid_shape = tuple(auto_capture_cfg["grid_shape"])
        self.samples_per_grid = int(auto_capture_cfg["samples_per_grid"])
        self.delay_between_captures = float(
            auto_capture_cfg.get("delay_between_captures", 1.0)
        )
        self.stability_frames = int(auto_capture_cfg.get("stability_frames", 5))
        self.stability_threshold = float(
            auto_capture_cfg.get("stability_threshold", 2.0)
        )
        self.stability_threshold_ratio = float(
            auto_capture_cfg.get("stability_threshold_ratio", 0.02)
        )
        self.min_total_samples = int(
            auto_capture_cfg.get(
                "min_total_samples",
                self.grid_shape[0] * self.grid_shape[1] * self.samples_per_grid,
            )
        )
        self.pose_novelty_center_distance_ratio = float(
            auto_capture_cfg.get("pose_novelty_center_distance_ratio", 0.08)
        )
        self.pose_novelty_area_delta = float(
            auto_capture_cfg.get("pose_novelty_area_delta", 0.02)
        )
        self.pose_novelty_aspect_delta = float(
            auto_capture_cfg.get("pose_novelty_aspect_delta", 0.12)
        )
        self.reset()

    def reset(self):
        self.grid_coverage = np.zeros(self.grid_shape, dtype=int)
        self.stability_counter = 0
        self.last_corners_center = None
        self.last_capture_time = 0.0
        self.last_motion_px = None
        self.last_effective_stability_threshold_px = float(self.stability_threshold)
        self.last_bbox_diagonal_px = None
        self.objpoints = []
        self.imgpoints = []
        self.sample_records = []

    @property
    def sample_count(self):
        return len(self.objpoints)

    @property
    def coverage_cell_count(self):
        return int(np.count_nonzero(self.grid_coverage > 0))

    @property
    def coverage_target_cell_count(self):
        return int(self.grid_coverage.size)

    @property
    def remaining_required_samples(self):
        return max(int(self.min_total_samples) - int(self.sample_count), 0)

    @property
    def remaining_coverage_cells(self):
        return int(np.count_nonzero(self.grid_coverage < self.samples_per_grid))

    def _cell_label(self, cell_y, cell_x):
        row_names = ["top", "middle", "bottom"]
        col_names = ["left", "center", "right"]
        if len(row_names) == self.grid_shape[0] and len(col_names) == self.grid_shape[1]:
            return f"{row_names[cell_y]}-{col_names[cell_x]}"
        return f"r{int(cell_y) + 1}c{int(cell_x) + 1}"

    def coverage_heatmap_snapshot(self):
        rows, cols = self.grid_shape
        cells = []
        for cell_y in range(rows):
            for cell_x in range(cols):
                cells.append(
                    {
                        "x": int(cell_x),
                        "y": int(cell_y),
                        "count": int(self.grid_coverage[cell_y, cell_x]),
                        "label": self._cell_label(cell_y, cell_x),
                    }
                )
        cells.sort(key=lambda item: (int(item["count"]), int(item["y"]), int(item["x"])))
        return cells

    def coverage_complete(self):
        return bool(
            self.grid_coverage.size > 0
            and np.all(self.grid_coverage >= self.samples_per_grid)
        )

    def sample_target_met(self):
        return bool(self.sample_count >= self.min_total_samples)

    def pose_diversity_snapshot(self):
        target_area_span = max(float(self.pose_novelty_area_delta) * 2.0, 0.08)
        target_aspect_span = max(float(self.pose_novelty_aspect_delta) * 2.0, 0.24)
        if not self.sample_records:
            return {
                "area_span": 0.0,
                "aspect_span": 0.0,
                "center_span_x": 0.0,
                "center_span_y": 0.0,
                "scale_diversity_ready": False,
                "tilt_diversity_ready": False,
                "target_area_span": float(target_area_span),
                "target_aspect_span": float(target_aspect_span),
            }

        pose_summaries = [
            self._pose_summary_from_record(sample_record)
            for sample_record in self.sample_records
        ]
        area_values = [float(item.get("bbox_area_ratio", 0.0)) for item in pose_summaries]
        aspect_values = [float(item.get("bbox_aspect_ratio", 0.0)) for item in pose_summaries]
        center_x_values = [
            float((item.get("center_xy_normalized") or {}).get("x", 0.0))
            for item in pose_summaries
        ]
        center_y_values = [
            float((item.get("center_xy_normalized") or {}).get("y", 0.0))
            for item in pose_summaries
        ]
        area_span = float(max(area_values) - min(area_values)) if area_values else 0.0
        aspect_span = float(max(aspect_values) - min(aspect_values)) if aspect_values else 0.0
        center_span_x = (
            float(max(center_x_values) - min(center_x_values)) if center_x_values else 0.0
        )
        center_span_y = (
            float(max(center_y_values) - min(center_y_values)) if center_y_values else 0.0
        )
        return {
            "area_span": float(area_span),
            "aspect_span": float(aspect_span),
            "center_span_x": float(center_span_x),
            "center_span_y": float(center_span_y),
            "scale_diversity_ready": bool(area_span >= target_area_span),
            "tilt_diversity_ready": bool(aspect_span >= target_aspect_span),
            "target_area_span": float(target_area_span),
            "target_aspect_span": float(target_aspect_span),
        }

    def _build_operator_guidance(
        self,
        stage,
        remaining_coverage_cells,
        remaining_samples,
        diversity,
        sparsest_cells,
    ):
        if stage == "collect_coverage":
            sparsest_preview = ", ".join(
                f"{cell['label']}={cell['count']}" for cell in sparsest_cells[:3]
            )
            guidance_actions = [
                (
                    "Target the least-covered cells first: "
                    f"{sparsest_preview}."
                    if sparsest_preview
                    else f"Shift the board into new image regions; {remaining_coverage_cells} coverage cells are still empty."
                ),
                "Keep the full board visible and avoid cropping at the image edge.",
            ]
            return {
                "guidance_summary": "Stage 1: fill the remaining image regions first.",
                "guidance_actions": guidance_actions,
            }

        if stage == "collect_diverse_samples":
            guidance_actions = []
            if not bool(diversity.get("scale_diversity_ready")):
                guidance_actions.append(
                    "Add a closer or farther view so the board size changes clearly in the image (about 15-25%)."
                )
            if not bool(diversity.get("tilt_diversity_ready")):
                guidance_actions.append(
                    "Add an oblique view by tilting the board left/right or up/down (about 10-20 degrees)."
                )
            if not guidance_actions:
                guidance_actions.append(
                    "Use a clearly different pose; simple translation is no longer enough after coverage is full."
                )
            guidance_actions.append(
                "Hold each new pose steady until it is accepted and keep the full board visible."
            )
            return {
                "guidance_summary": (
                    f"Stage 2: coverage is complete; collect {remaining_samples} more novel poses."
                ),
                "guidance_actions": guidance_actions,
            }

        return {
            "guidance_summary": "Coverage and sample target met; calibration can proceed.",
            "guidance_actions": [
                "Capture extra samples only if you want redundancy before solving.",
            ],
        }

    def progress_snapshot(self):
        coverage_complete = self.coverage_complete()
        sample_target_met = self.sample_target_met()
        if not coverage_complete:
            stage = "collect_coverage"
        elif not sample_target_met:
            stage = "collect_diverse_samples"
        else:
            stage = "ready_to_calibrate"
        diversity = self.pose_diversity_snapshot()
        heatmap = self.coverage_heatmap_snapshot()
        sparsest_cells = [
            dict(cell)
            for cell in heatmap
            if int(cell.get("count", 0)) < int(self.samples_per_grid)
        ]
        progress = {
            "stage": stage,
            "coverage_complete": bool(coverage_complete),
            "sample_target_met": bool(sample_target_met),
            "coverage_cell_count": int(self.coverage_cell_count),
            "coverage_target_cell_count": int(self.coverage_target_cell_count),
            "coverage_heatmap": heatmap,
            "remaining_coverage_cells": int(self.remaining_coverage_cells),
            "sample_count": int(self.sample_count),
            "required_sample_count": int(self.min_total_samples),
            "remaining_required_samples": int(self.remaining_required_samples),
            "pose_diversity": diversity,
            "sparsest_cells": sparsest_cells,
        }
        progress.update(
            self._build_operator_guidance(
                stage,
                int(progress["remaining_coverage_cells"]),
                int(progress["remaining_required_samples"]),
                diversity,
                sparsest_cells,
            )
        )
        return progress

    def can_capture_now(self, now_s=None):
        current_time = time.time() if now_s is None else float(now_s)
        return (current_time - self.last_capture_time) >= self.delay_between_captures

    def note_detection(self, image_points):
        corners = np.asarray(image_points, dtype=float).reshape(-1, 2)
        center = np.mean(corners, axis=0)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        bbox_size = np.maximum(bbox_max - bbox_min, 0.0)
        bbox_diagonal = float(np.linalg.norm(bbox_size))
        effective_threshold = max(
            float(self.stability_threshold),
            float(bbox_diagonal) * max(float(self.stability_threshold_ratio), 0.0),
        )
        motion = None
        if self.last_corners_center is not None:
            motion = float(np.linalg.norm(center - self.last_corners_center))
            if motion < effective_threshold:
                self.stability_counter += 1
            else:
                self.stability_counter = 0
        self.last_corners_center = center
        self.last_motion_px = motion
        self.last_effective_stability_threshold_px = float(effective_threshold)
        self.last_bbox_diagonal_px = float(bbox_diagonal)
        return {
            "center": center,
            "motion_px": motion,
            "bbox_diagonal_px": float(bbox_diagonal),
            "effective_threshold_px": float(effective_threshold),
            "stability_counter": int(self.stability_counter),
        }

    def reset_stability(self):
        self.stability_counter = 0

    def _grid_index(self, coord, size, cells):
        cells = max(int(cells), 1)
        size = max(float(size), 1.0)
        clipped = min(max(float(coord), 0.0), max(size - 1e-6, 0.0))
        return min(cells - 1, max(0, int((clipped / size) * cells)))

    def occupied_grid_cells(self, refined, image_size_wh):
        width, height = int(image_size_wh[0]), int(image_size_wh[1])
        corners = np.asarray(refined, dtype=float).reshape(-1, 2)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        rows, cols = self.grid_shape
        min_x = self._grid_index(bbox_min[0], width, cols)
        max_x = self._grid_index(bbox_max[0], width, cols)
        min_y = self._grid_index(bbox_min[1], height, rows)
        max_y = self._grid_index(bbox_max[1], height, rows)
        occupied = []
        for cell_y in range(min_y, max_y + 1):
            for cell_x in range(min_x, max_x + 1):
                occupied.append((int(cell_x), int(cell_y)))
        if occupied:
            return occupied
        center = np.mean(corners, axis=0)
        return [
            (
                self._grid_index(center[0], width, cols),
                self._grid_index(center[1], height, rows),
            )
        ]

    def cells_need_samples(self, occupied_cells):
        return any(
            self.grid_coverage[cell_y, cell_x] < self.samples_per_grid
            for cell_x, cell_y in occupied_cells
        )

    def _pose_summary(self, refined, image_size_wh):
        width, height = int(image_size_wh[0]), int(image_size_wh[1])
        corners = np.asarray(refined, dtype=float).reshape(-1, 2)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        center = np.mean(corners, axis=0)
        span_x = float(max(bbox_max[0] - bbox_min[0], 0.0))
        span_y = float(max(bbox_max[1] - bbox_min[1], 0.0))
        return {
            "center_xy_normalized": {
                "x": float(center[0] / max(width, 1)),
                "y": float(center[1] / max(height, 1)),
            },
            "bbox_area_ratio": float(span_x * span_y / max(width * height, 1)),
            "bbox_aspect_ratio": float(span_x / max(span_y, 1e-6)),
        }

    def _pose_summary_from_record(self, sample_record):
        pose_summary = dict(sample_record.get("pose_summary") or {})
        if pose_summary:
            return pose_summary
        bbox = sample_record.get("image_bbox") or {}
        center = dict(bbox.get("center_xy_normalized") or {})
        min_xy = bbox.get("min_xy_px") or {}
        max_xy = bbox.get("max_xy_px") or {}
        span_x = float(max(float(max_xy.get("x", 0.0)) - float(min_xy.get("x", 0.0)), 0.0))
        span_y = float(max(float(max_xy.get("y", 0.0)) - float(min_xy.get("y", 0.0)), 0.0))
        return {
            "center_xy_normalized": {
                "x": float(center.get("x", 0.0)),
                "y": float(center.get("y", 0.0)),
            },
            "bbox_area_ratio": float(bbox.get("bbox_area_ratio", 0.0)),
            "bbox_aspect_ratio": float(span_x / max(span_y, 1e-6)),
        }

    def _pose_novelty_check(self, pose_summary):
        closest_sample_id = None
        closest_center_distance = None
        closest_area_delta = None
        closest_aspect_delta = None
        for sample_record in self.sample_records:
            previous_pose = self._pose_summary_from_record(sample_record)
            center_x = float(pose_summary["center_xy_normalized"]["x"])
            center_y = float(pose_summary["center_xy_normalized"]["y"])
            prev_x = float(previous_pose["center_xy_normalized"]["x"])
            prev_y = float(previous_pose["center_xy_normalized"]["y"])
            center_distance = float(
                np.linalg.norm(np.asarray([center_x - prev_x, center_y - prev_y], dtype=float))
            )
            area_delta = abs(
                float(pose_summary["bbox_area_ratio"])
                - float(previous_pose["bbox_area_ratio"])
            )
            aspect_delta = abs(
                float(pose_summary["bbox_aspect_ratio"])
                - float(previous_pose["bbox_aspect_ratio"])
            )
            if closest_center_distance is None or center_distance < closest_center_distance:
                closest_sample_id = int(sample_record.get("sample_id", 0))
                closest_center_distance = float(center_distance)
                closest_area_delta = float(area_delta)
                closest_aspect_delta = float(aspect_delta)
            similar_pose = (
                center_distance < self.pose_novelty_center_distance_ratio
                and area_delta < self.pose_novelty_area_delta
                and aspect_delta < self.pose_novelty_aspect_delta
            )
            if similar_pose:
                return {
                    "accept": False,
                    "capture_reason": "pose_not_novel",
                    "closest_sample_id": closest_sample_id,
                    "closest_center_distance_ratio": float(center_distance),
                    "closest_area_delta": float(area_delta),
                    "closest_aspect_delta": float(aspect_delta),
                }
        return {
            "accept": True,
            "capture_reason": "pose_novel",
            "closest_sample_id": closest_sample_id,
            "closest_center_distance_ratio": closest_center_distance,
            "closest_area_delta": closest_area_delta,
            "closest_aspect_delta": closest_aspect_delta,
        }

    def evaluate_capture_candidate(self, refined, image_size_wh):
        occupied_cells = self.occupied_grid_cells(refined, image_size_wh)
        coverage_complete = self.coverage_complete()
        remaining_samples = int(self.remaining_required_samples)
        decision = {
            "occupied_grid_cells": [
                {"x": int(cell_x), "y": int(cell_y)}
                for cell_x, cell_y in occupied_cells
            ],
            "coverage_complete": bool(coverage_complete),
            "remaining_required_samples": int(remaining_samples),
        }
        if not coverage_complete:
            accept = self.cells_need_samples(occupied_cells)
            decision.update(
                {
                    "accept": bool(accept),
                    "capture_reason": (
                        "coverage_needed" if accept else "move_to_uncovered_cells"
                    ),
                }
            )
            return decision
        if self.sample_count >= self.min_total_samples:
            decision.update(
                {
                    "accept": False,
                    "capture_reason": "sample_target_met",
                }
            )
            return decision
        pose_summary = self._pose_summary(refined, image_size_wh)
        decision["pose_summary"] = pose_summary
        decision.update(self._pose_novelty_check(pose_summary))
        return decision

    def record_grid_coverage(self, sample_record):
        occupied_cells = sample_record.get("occupied_grid_cells") or []
        if not occupied_cells:
            grid_cell = sample_record.get("grid_cell") or {}
            occupied_cells = [
                {
                    "x": int(grid_cell.get("x", 0)),
                    "y": int(grid_cell.get("y", 0)),
                }
            ]
        for cell in occupied_cells:
            cell_x = int(cell.get("x", 0))
            cell_y = int(cell.get("y", 0))
            self.grid_coverage[cell_y, cell_x] += 1

    def build_sample_record(
        self,
        refined,
        image_size_wh,
        *,
        sample_id=None,
        source,
        source_path=None,
        grid_cell=None,
    ):
        width, height = int(image_size_wh[0]), int(image_size_wh[1])
        corners = np.asarray(refined, dtype=float).reshape(-1, 2)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        center = np.mean(corners, axis=0)
        occupied_grid_cells = self.occupied_grid_cells(refined, (width, height))
        rows, cols = self.grid_shape
        if grid_cell is None:
            cell_x = self._grid_index(center[0], width, cols)
            cell_y = self._grid_index(center[1], height, rows)
        else:
            cell_x = int(grid_cell[0])
            cell_y = int(grid_cell[1])
        return {
            "sample_id": int(self.sample_count + 1 if sample_id is None else sample_id),
            "source": str(source),
            "source_path": source_path,
            "grid_cell": {"x": cell_x, "y": cell_y},
            "occupied_grid_cells": [
                {"x": int(occupied_x), "y": int(occupied_y)}
                for occupied_x, occupied_y in occupied_grid_cells
            ],
            "image_size_wh": {"width": width, "height": height},
            "pose_summary": self._pose_summary(refined, (width, height)),
            "image_bbox": {
                "min_xy_px": {"x": float(bbox_min[0]), "y": float(bbox_min[1])},
                "max_xy_px": {"x": float(bbox_max[0]), "y": float(bbox_max[1])},
                "center_xy_normalized": {
                    "x": float(center[0] / max(width, 1)),
                    "y": float(center[1] / max(height, 1)),
                },
                "edge_margin_px": float(
                    min(
                        bbox_min[0],
                        bbox_min[1],
                        max(width - bbox_max[0], 0.0),
                        max(height - bbox_max[1], 0.0),
                    )
                ),
                "bbox_area_ratio": float(
                    max((bbox_max[0] - bbox_min[0]), 0.0)
                    * max((bbox_max[1] - bbox_min[1]), 0.0)
                    / max(width * height, 1)
                ),
            },
        }

    def append_sample(
        self,
        refined,
        image_size_wh,
        *,
        object_points,
        source,
        source_path=None,
        grid_cell=None,
    ):
        sample_id = self.sample_count + 1
        sample_record = self.build_sample_record(
            refined,
            image_size_wh,
            sample_id=sample_id,
            source=source,
            source_path=source_path,
            grid_cell=grid_cell,
        )
        self.objpoints.append(np.asarray(object_points, dtype=np.float32).copy())
        self.imgpoints.append(np.asarray(refined, dtype=np.float32).copy())
        self.sample_records.append(sample_record)
        self.record_grid_coverage(sample_record)
        self.last_capture_time = time.time()
        return sample_record
