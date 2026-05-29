from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from camera2camera.metrics import transform_delta_metrics
from camera2camera.reference_pipeline import (
    _corner_permutations,
    _image_point_metrics,
    _load_camera_config,
    _medoid_transform,
    _solve_board_pose,
    run_reference_calibration_from_payload,
)
from camera.intrinsic_capture import (
    apply_capture_settings,
    build_capture_runtime_info,
    display_size,
    is_visually_empty_frame,
    open_managed_capture,
)
from camera.intrinsic_sampling import IntrinsicSamplingState
from camera.intrinsic_targets import CalibrationTargetDetector
from camera.intrinsic_visualization import (
    build_stereo_comparison_canvas,
    draw_aprilgrid_debug,
    draw_dynamic_ui,
    draw_text,
    render_preserving_aspect_ratio,
)
from lidar2lidar.extrinsic_io import load_extrinsics_file


def _slugify(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "session"


@dataclass(frozen=True)
class StereoCaptureSessionPaths:
    label: str
    session_dir: Path
    parent_dir: Path
    child_dir: Path
    debug_dir: Path
    provisional_dir: Path
    manifest_path: Path


def _resolve_path(value: str | None, *, base_directory: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_directory / path).resolve()
    return str(path)


def _build_target_detector(
    payload: dict[str, Any],
) -> tuple[str, CalibrationTargetDetector]:
    target_payload = copy.deepcopy(payload.get("target", {}) or {})
    target_type = str(target_payload.get("type", "checkerboard")).lower()
    detector_target_type = (
        "chessboard" if target_type == "checkerboard" else target_type
    )
    detector_cfg: dict[str, Any] = {"target_type": detector_target_type}
    pattern_size = None
    square_size = None
    if target_type == "checkerboard":
        pattern_size = tuple(target_payload.get("pattern_size", [11, 8]))
        square_size = float(target_payload.get("square_size_m", 0.025))
        detector_cfg["pattern_size"] = list(pattern_size)
        detector_cfg["square_size"] = float(square_size)
    elif target_type == "aprilgrid":
        detector_cfg["aprilgrid"] = copy.deepcopy(
            target_payload.get("aprilgrid") or target_payload
        )
    elif target_type == "charuco":
        detector_cfg["charuco"] = copy.deepcopy(
            target_payload.get("charuco") or target_payload
        )
    else:
        raise ValueError(f"Unsupported camera2camera live target_type: {target_type}")
    return target_type, CalibrationTargetDetector(
        detector_cfg,
        pattern_size=pattern_size,
        square_size=square_size,
    )


def _resolve_camera_source(
    camera_payload: dict[str, Any]
) -> tuple[str | int, dict[str, Any]]:
    source_cfg = dict(camera_payload.get("source") or {})
    source = source_cfg.get("uri")
    if source not in (None, ""):
        return str(source), {
            "selected_camera_index": None,
            "camera_config": source_cfg,
            "source_type": "network",
        }
    device_index = source_cfg.get("device_index", source_cfg.get("camera_index", 0))
    if isinstance(device_index, str):
        stripped = device_index.strip()
        if stripped.isdigit():
            device_index = int(stripped)
    return int(device_index), {
        "selected_camera_index": None,
        "camera_config": source_cfg,
        "source_type": "device",
    }


def _annotate_detection(frame: np.ndarray, detection) -> np.ndarray:
    display = frame.copy()
    if detection is None or not bool(getattr(detection, "found", False)):
        return display
    points = np.asarray(detection.image_points, dtype=float).reshape(-1, 2)
    for point in points[: min(len(points), 400)]:
        cv2.circle(
            display,
            (int(round(point[0])), int(round(point[1]))),
            2,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    cv2.rectangle(
        display,
        (int(round(bbox_min[0])), int(round(bbox_min[1]))),
        (int(round(bbox_max[0])), int(round(bbox_max[1]))),
        (0, 200, 255),
        2,
    )
    return display


class StereoLiveCapture:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        base_directory: str | Path | None = None,
        session_name: str | None = None,
        capture_only: bool = False,
        require_release_ready: bool | None = None,
        output_dir_override: str | None = None,
    ):
        self.cfg = copy.deepcopy(payload)
        self.base_directory = (
            Path(base_directory).expanduser().resolve()
            if base_directory is not None
            else Path.cwd()
        )
        self.capture_only = bool(capture_only)
        cfg_require_release_ready = bool(self.cfg.get("require_release_ready", False))
        self.require_release_ready = (
            cfg_require_release_ready
            if require_release_ready is None
            else bool(require_release_ready)
        )
        self.live_cfg = self.cfg.get("live_capture", {}) or {}
        self.workflow_cfg = self.cfg.get("workflow", {}) or {}
        self.capture_cfg = self.cfg.get("capture", {}) or {}
        self.auto_capture_cfg = self.cfg.get("auto_capture_settings", {}) or {}
        self.extraction_cfg = self.cfg.get("extraction", {}) or {}
        self.output_cfg = self.cfg.get("output", {}) or {}
        self.session_label = _slugify(
            session_name
            or self.workflow_cfg.get("session_name")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        workflow_root = _resolve_path(
            str(self.workflow_cfg.get("root_dir", "outputs/camera2camera")),
            base_directory=self.base_directory,
        )
        self.workflow_root = Path(str(workflow_root)).resolve()
        self.session = self._prepare_capture_session()
        self.output_dir = self._resolve_output_dir(output_dir_override)

        cameras = self.cfg.get("cameras", {}) or {}
        self.parent_camera_payload = copy.deepcopy(cameras.get("parent", {}) or {})
        self.child_camera_payload = copy.deepcopy(cameras.get("child", {}) or {})
        for camera_payload in (self.parent_camera_payload, self.child_camera_payload):
            resolved_intrinsics = _resolve_path(
                camera_payload.get("intrinsics_path"),
                base_directory=self.base_directory,
            )
            if resolved_intrinsics is not None:
                camera_payload["intrinsics_path"] = resolved_intrinsics

        self.target_type, self.parent_detector = _build_target_detector(self.cfg)
        _target_type, self.child_detector = _build_target_detector(self.cfg)
        if _target_type != self.target_type:
            raise ValueError(
                "Parent and child target detectors must use the same target."
            )
        (
            self.parent_camera_matrix,
            self.parent_distortion,
            self.parent_camera_metadata,
        ) = _load_camera_config(
            self.parent_camera_payload,
            base_directory=self.base_directory,
        )
        (
            self.child_camera_matrix,
            self.child_distortion,
            self.child_camera_metadata,
        ) = _load_camera_config(
            self.child_camera_payload,
            base_directory=self.base_directory,
        )
        self.initial_relative_transform = None
        if self.cfg.get("initial_transform_path"):
            initial_transform_path = _resolve_path(
                str(self.cfg.get("initial_transform_path")),
                base_directory=self.base_directory,
            )
            if initial_transform_path is not None:
                (
                    self.initial_relative_transform,
                    _parent_frame,
                    _child_frame,
                    _translation,
                    _rotation,
                ) = load_extrinsics_file(initial_transform_path)

        self.parent_sampling = IntrinsicSamplingState(self.auto_capture_cfg)
        self.child_sampling = IntrinsicSamplingState(self.auto_capture_cfg)
        self.parent_source, self.parent_source_meta = _resolve_camera_source(
            self.parent_camera_payload
        )
        self.child_source, self.child_source_meta = _resolve_camera_source(
            self.child_camera_payload
        )
        self.parent_capture = None
        self.child_capture = None
        self.parent_runtime_info = None
        self.child_runtime_info = None
        self.parent_last_debug = None
        self.child_last_debug = None
        self.feedback_text = "Searching for a shared calibration target..."
        self.last_provisional_result: dict[str, Any] | None = None
        self.last_final_result: dict[str, Any] | None = None
        self.frame_counter = 0
        self.window_name = "Camera2Camera Live Capture"
        self.provisional_eval_interval = max(
            1, int(self.live_cfg.get("provisional_eval_interval", 1))
        )
        self.auto_stop_on_release_ready = bool(
            self.live_cfg.get("auto_stop_on_release_ready", True)
        )
        self.min_bbox_area_ratio = float(
            self.extraction_cfg.get("min_bbox_area_ratio", 0.003)
        )
        self.min_edge_margin_px = float(
            self.extraction_cfg.get("min_edge_margin_px", 16.0)
        )
        self.max_pnp_reprojection_rms_px = float(
            self.extraction_cfg.get("max_pnp_reprojection_rms_px", 1.5)
        )
        self.max_candidate_translation_delta_m = float(
            self.extraction_cfg.get("max_candidate_translation_delta_m", 0.25)
        )
        self.max_candidate_rotation_delta_deg = float(
            self.extraction_cfg.get("max_candidate_rotation_delta_deg", 3.0)
        )
        self.accepted_relative_transforms: list[np.ndarray] = []
        self.provisional_consensus_transform: np.ndarray | None = None
        self.final_consensus_transform: np.ndarray | None = None
        self.last_pair_review: dict[str, Any] | None = None
        self.current_diagnostics = {
            "parent": "Waiting for target detection.",
            "child": "Waiting for target detection.",
            "stereo": "Waiting for a valid stereo pair.",
        }

    @property
    def sample_count(self) -> int:
        return int(
            min(self.parent_sampling.sample_count, self.child_sampling.sample_count)
        )

    @property
    def required_sample_count(self) -> int:
        return int(
            max(self.parent_sampling.min_total_samples, self._minimum_pair_count())
        )

    def _minimum_pair_count(self) -> int:
        optimization_cfg = self.cfg.get("optimization", {}) or {}
        return int(optimization_cfg.get("min_pairs", 8))

    def _prepare_capture_session(self) -> StereoCaptureSessionPaths:
        session_dir = self.workflow_root / "captures" / self.session_label
        parent_dir = session_dir / "parent"
        child_dir = session_dir / "child"
        debug_dir = session_dir / "debug"
        provisional_dir = session_dir / "provisional"
        for path in (parent_dir, child_dir, debug_dir, provisional_dir):
            path.mkdir(parents=True, exist_ok=True)
        return StereoCaptureSessionPaths(
            label=self.session_label,
            session_dir=session_dir,
            parent_dir=parent_dir,
            child_dir=child_dir,
            debug_dir=debug_dir,
            provisional_dir=provisional_dir,
            manifest_path=session_dir / "capture_session.yaml",
        )

    def _resolve_output_dir(self, output_dir_override: str | None) -> Path:
        if output_dir_override:
            return Path(
                _resolve_path(output_dir_override, base_directory=self.base_directory)
            ).resolve()
        configured_output = self.output_cfg.get("directory")
        if configured_output:
            configured_output = str(configured_output)
            if configured_output != "outputs/camera2camera/reference":
                return Path(
                    _resolve_path(configured_output, base_directory=self.base_directory)
                ).resolve()
        return (self.workflow_root / "runs" / self.session_label).resolve()

    def _build_capture_manifest(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "session_label": self.session.label,
            "capture_root": str(self.session.session_dir),
            "parent_images_dir": str(self.session.parent_dir),
            "child_images_dir": str(self.session.child_dir),
            "target_type": self.target_type,
            "sample_count": int(self.sample_count),
            "required_sample_count": int(self.required_sample_count),
            "capture_only": bool(self.capture_only),
            "require_release_ready": bool(self.require_release_ready),
            "parent_progress": self.parent_sampling.progress_snapshot(),
            "child_progress": self.child_sampling.progress_snapshot(),
            "feedback_text": str(self.feedback_text),
            "current_diagnostics": copy.deepcopy(self.current_diagnostics),
            "last_pair_review": copy.deepcopy(self.last_pair_review),
            "last_provisional_result": copy.deepcopy(self.last_provisional_result),
            "last_final_result": copy.deepcopy(self.last_final_result),
            "final_output_dir": str(self.output_dir),
            "sources": {
                "parent": {
                    "source": str(self.parent_source),
                    "source_type": self.parent_source_meta.get("source_type"),
                },
                "child": {
                    "source": str(self.child_source),
                    "source_type": self.child_source_meta.get("source_type"),
                },
            },
            "runtime_info": {
                "parent": copy.deepcopy(self.parent_runtime_info),
                "child": copy.deepcopy(self.child_runtime_info),
            },
        }

    def _write_capture_manifest(self, status: str) -> None:
        with self.session.manifest_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self._build_capture_manifest(status), file, sort_keys=False)

    def _open_captures(self) -> bool:
        self.parent_capture, _ = open_managed_capture(
            self.parent_source,
            self.cfg,
            self.parent_source_meta,
        )
        if self.parent_capture is None:
            print("[ERROR] Cannot open parent capture source", self.parent_source)
            return False
        self.child_capture, _ = open_managed_capture(
            self.child_source,
            self.cfg,
            self.child_source_meta,
        )
        if self.child_capture is None:
            print("[ERROR] Cannot open child capture source", self.child_source)
            self.parent_capture.release()
            self.parent_capture = None
            return False
        apply_capture_settings(self.parent_capture, self.cfg, self.parent_source_meta)
        apply_capture_settings(self.child_capture, self.cfg, self.child_source_meta)
        return True

    def _close_captures(self) -> None:
        if self.parent_capture is not None:
            self.parent_capture.release()
            self.parent_capture = None
        if self.child_capture is not None:
            self.child_capture.release()
            self.child_capture = None

    def _read_frame(self, capture, name: str) -> np.ndarray | None:
        if capture is None:
            return None
        ret, frame = capture.read()
        if not ret or frame is None or frame.size == 0:
            if getattr(capture, "managed_capture", False) and hasattr(
                capture, "is_ready"
            ):
                if not capture.is_ready():
                    return None
            print(f"[WARN] Failed to read {name} frame; continuing...")
            return None
        if is_visually_empty_frame(frame):
            print(f"[WARN] {name} frame appears invalid/blank; continuing...")
            return None
        return frame

    def _detect(self, frame: np.ndarray, detector: CalibrationTargetDetector):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection = detector.detect(
            gray,
            self.frame_counter,
            {"resize_factor": 1.0, "detection_interval": 1},
        )
        return gray, detection

    def _draw_live_panel(
        self, title: str, frame: np.ndarray, detection, sampling
    ) -> np.ndarray:
        display = _annotate_detection(frame, detection)
        panel_key = title.lower()
        draw_dynamic_ui(
            display,
            sampling.grid_coverage,
            sampling.grid_shape,
            self.feedback_text,
            sampling.progress_snapshot(),
        )
        if detection is not None and bool(getattr(detection, "found", False)):
            draw_text(display, title, (40, 40), color=(255, 255, 255), scale=0.8)
        else:
            draw_text(
                display,
                f"{title}: target lost",
                (40, 40),
                color=(0, 200, 255),
                scale=0.8,
            )
        draw_text(
            display,
            str(self.current_diagnostics.get(panel_key, "")),
            (40, 75),
            color=(180, 240, 255),
            scale=0.6,
        )
        debug_info = getattr(detection, "debug_info", None)
        if isinstance(debug_info, dict):
            draw_aprilgrid_debug(display, debug_info)
        return display

    def _pair_rejection_feedback(
        self,
        parent_decision: dict[str, Any] | None,
        child_decision: dict[str, Any] | None,
    ) -> str:
        reason_map = {
            "move_to_uncovered_cells": "Move the board into uncovered image regions.",
            "pose_not_novel": (
                "Change depth/tilt; current pose is too similar " "to accepted pairs."
            ),
            "sample_target_met": (
                "Sample target met; waiting for stereo quality review."
            ),
        }
        messages = []
        for camera_name, decision in (
            ("parent", parent_decision),
            ("child", child_decision),
        ):
            if not decision or bool(decision.get("accept")):
                continue
            capture_reason = str(decision.get("capture_reason", ""))
            detail = reason_map.get(capture_reason, capture_reason or "not accepted")
            messages.append(f"{camera_name}: {detail}")
        if messages:
            return " | ".join(messages)
        return "Hold both cameras steady on a new shared pose."

    def _current_consensus_transform(self) -> np.ndarray | None:
        if self.final_consensus_transform is not None:
            return np.asarray(self.final_consensus_transform, dtype=float)
        if self.provisional_consensus_transform is not None:
            return np.asarray(self.provisional_consensus_transform, dtype=float)
        if self.accepted_relative_transforms:
            return _medoid_transform(self.accepted_relative_transforms)
        if self.initial_relative_transform is not None:
            return np.asarray(self.initial_relative_transform, dtype=float)
        return None

    def _set_diagnostics(self, *, parent: str, child: str, stereo: str) -> None:
        self.current_diagnostics = {
            "parent": str(parent),
            "child": str(child),
            "stereo": str(stereo),
        }

    def _missing_target_feedback(self, camera_name: str) -> str:
        target_label = (
            "AprilGrid" if self.target_type == "aprilgrid" else self.target_type
        )
        return (
            f"{camera_name}: {target_label} not detected. Keep the full board visible, "
            "reduce glare/blur, and move slightly closer."
        )

    def _camera_pose_summary(
        self, pose: dict[str, Any], metrics: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "reprojection_rms_px": float(pose["reprojection_rms_px"]),
            "board_center_camera_m": pose["board_center_camera_m"],
            "board_tilt_deg": float(pose["board_tilt_deg"]),
            "bbox_area_ratio": float(metrics["bbox_area_ratio"]),
            "edge_margin_px": float(metrics["edge_margin_px"]),
            "center_xy_normalized": metrics["center_xy_normalized"],
        }

    def _transform_summary(self, transform: np.ndarray) -> dict[str, Any]:
        matrix = np.asarray(transform, dtype=float)
        euler_deg = R.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
        return {
            "translation_m": {
                "x": float(matrix[0, 3]),
                "y": float(matrix[1, 3]),
                "z": float(matrix[2, 3]),
            },
            "euler_deg": {
                "roll": float(euler_deg[0]),
                "pitch": float(euler_deg[1]),
                "yaw": float(euler_deg[2]),
            },
        }

    def _next_collection_action(self) -> str:
        parent_progress = self.parent_sampling.progress_snapshot()
        child_progress = self.child_sampling.progress_snapshot()
        if (
            parent_progress["coverage_cell_count"]
            < parent_progress["coverage_target_cell_count"]
            or child_progress["coverage_cell_count"]
            < child_progress["coverage_target_cell_count"]
        ):
            return "Move the board into image regions that are still uncovered."
        if (
            parent_progress["stage"] == "collect_diverse_samples"
            or child_progress["stage"] == "collect_diverse_samples"
        ):
            return (
                "Add more depth, tilt, and left-right variation instead of "
                "repeating the same pose."
            )
        remaining_pairs = max(self._minimum_pair_count() - self.sample_count, 0)
        if remaining_pairs > 0:
            return (
                f"Collect {remaining_pairs} more high-quality stereo pairs before "
                "full stereo review."
            )
        return (
            "Dataset is large enough; use provisional/final stereo review to "
            "judge release readiness."
        )

    def _write_pair_review(self) -> None:
        if self.last_pair_review is None:
            return
        review_path = (
            self.session.debug_dir / f"sample_{self.sample_count:03d}_review.yaml"
        )
        with review_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.last_pair_review, file, sort_keys=False)

    def _evaluate_candidate_pair(
        self,
        parent_detection,
        child_detection,
        *,
        parent_size: tuple[int, int],
        child_size: tuple[int, int],
    ) -> dict[str, Any]:
        parent_image_points = np.asarray(
            parent_detection.image_points, dtype=float
        ).reshape(-1, 2)
        child_image_points = np.asarray(
            child_detection.image_points, dtype=float
        ).reshape(-1, 2)
        parent_metrics = _image_point_metrics(parent_image_points, parent_size)
        child_metrics = _image_point_metrics(child_image_points, child_size)
        if parent_metrics["bbox_area_ratio"] < self.min_bbox_area_ratio:
            return {
                "accept": False,
                "reason": "parent_bbox_too_small",
                "feedback": "Parent board is too small in the image; move it closer.",
            }
        if child_metrics["bbox_area_ratio"] < self.min_bbox_area_ratio:
            return {
                "accept": False,
                "reason": "child_bbox_too_small",
                "feedback": "Child board is too small in the image; move it closer.",
            }
        if parent_metrics["edge_margin_px"] < self.min_edge_margin_px:
            return {
                "accept": False,
                "reason": "parent_near_edge",
                "feedback": "Move the board away from the parent image border.",
            }
        if child_metrics["edge_margin_px"] < self.min_edge_margin_px:
            return {
                "accept": False,
                "reason": "child_near_edge",
                "feedback": "Move the board away from the child image border.",
            }

        parent_pose = _solve_board_pose(
            np.asarray(parent_detection.object_points, dtype=float),
            parent_image_points,
            self.parent_camera_matrix,
            self.parent_distortion,
        )
        if parent_pose is None:
            return {
                "accept": False,
                "reason": "parent_pnp_failed",
                "feedback": (
                    "Parent pose solve failed; reduce blur and keep the board "
                    "fully visible."
                ),
            }
        if parent_pose["reprojection_rms_px"] > self.max_pnp_reprojection_rms_px:
            return {
                "accept": False,
                "reason": "parent_reprojection_high",
                "feedback": (
                    "Parent pose reprojection is too high; hold steadier, "
                    "improve focus, "
                    "and avoid grazing-angle views."
                ),
                "parent": self._camera_pose_summary(parent_pose, parent_metrics),
            }

        if self.target_type == "checkerboard":
            candidate_point_map = _corner_permutations(
                child_image_points,
                tuple(self.cfg.get("target", {}).get("pattern_size", [11, 8])),
            )
        else:
            candidate_point_map = {"identity": child_image_points}

        consensus_transform = self._current_consensus_transform()
        child_candidates = []
        for permutation, candidate_corners in candidate_point_map.items():
            child_pose = _solve_board_pose(
                np.asarray(child_detection.object_points, dtype=float),
                candidate_corners,
                self.child_camera_matrix,
                self.child_distortion,
            )
            if child_pose is None:
                continue
            if child_pose["reprojection_rms_px"] > self.max_pnp_reprojection_rms_px:
                continue
            relative_transform = child_pose["transform"] @ np.linalg.inv(
                parent_pose["transform"]
            )
            delta = None
            score = 0.0
            if consensus_transform is not None:
                delta = transform_delta_metrics(consensus_transform, relative_transform)
                score = delta["translation_norm_m"] / max(
                    self.max_candidate_translation_delta_m, 1e-6
                ) + delta["rotation_deg"] / max(
                    self.max_candidate_rotation_delta_deg, 1e-6
                )
            else:
                score = float(child_pose["reprojection_rms_px"])
            child_candidates.append(
                {
                    "permutation": permutation,
                    "corners": np.asarray(candidate_corners, dtype=float),
                    "child_pose": child_pose,
                    "relative_transform": relative_transform,
                    "delta_to_consensus": delta,
                    "score": float(score),
                }
            )

        if not child_candidates:
            return {
                "accept": False,
                "reason": "child_pnp_failed",
                "feedback": (
                    "Child pose solve failed; keep the board sharper, flatter, "
                    "and fully visible."
                ),
                "parent": self._camera_pose_summary(parent_pose, parent_metrics),
            }

        selected_child = min(child_candidates, key=lambda item: float(item["score"]))
        delta_to_consensus = selected_child.get("delta_to_consensus")
        consistency_gate_enabled = bool(
            (
                self.target_type != "checkerboard"
                and len(self.accepted_relative_transforms) >= 2
            )
            or self.provisional_consensus_transform is not None
            or self.final_consensus_transform is not None
        )
        if (
            consistency_gate_enabled
            and delta_to_consensus is not None
            and (
                float(delta_to_consensus["translation_norm_m"])
                > self.max_candidate_translation_delta_m
                or float(delta_to_consensus["rotation_deg"])
                > self.max_candidate_rotation_delta_deg
            )
        ):
            return {
                "accept": False,
                "reason": "inconsistent_relative_transform",
                "feedback": (
                    "This pair disagrees with the current stereo consensus; keep "
                    "both views "
                    "steady and avoid unsynchronized board motion."
                ),
                "parent": self._camera_pose_summary(parent_pose, parent_metrics),
                "child": self._camera_pose_summary(
                    selected_child["child_pose"], child_metrics
                ),
                "selected_child_permutation": selected_child["permutation"],
                "relative_transform_delta_to_consensus": delta_to_consensus,
            }

        return {
            "accept": True,
            "reason": None,
            "feedback": "Pair pose solve is good.",
            "parent": self._camera_pose_summary(parent_pose, parent_metrics),
            "child": self._camera_pose_summary(
                selected_child["child_pose"], child_metrics
            ),
            "selected_child_permutation": selected_child["permutation"],
            "relative_transform": np.asarray(
                selected_child["relative_transform"], dtype=float
            ),
            "relative_transform_delta_to_consensus": delta_to_consensus,
        }

    def _record_pair_review(self, pair_report: dict[str, Any]) -> None:
        remaining_pairs = max(self._minimum_pair_count() - self.sample_count, 0)
        self.last_pair_review = {
            "sample_index": int(self.sample_count),
            "target_type": self.target_type,
            "pair_pose_status": "accepted" if pair_report.get("accept") else "rejected",
            "parent": copy.deepcopy(pair_report.get("parent")),
            "child": copy.deepcopy(pair_report.get("child")),
            "relative_transform_summary": (
                self._transform_summary(pair_report["relative_transform"])
                if pair_report.get("relative_transform") is not None
                else None
            ),
            "selected_child_permutation": pair_report.get("selected_child_permutation"),
            "relative_transform_delta_to_consensus": copy.deepcopy(
                pair_report.get("relative_transform_delta_to_consensus")
            ),
            "remaining_pairs_to_full_review": int(remaining_pairs),
            "next_action": self._next_collection_action(),
            "parent_progress": self.parent_sampling.progress_snapshot(),
            "child_progress": self.child_sampling.progress_snapshot(),
        }
        self._write_pair_review()

    def _provisional_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.cfg)
        payload.setdefault("cameras", {})
        payload["cameras"].setdefault("parent", {})
        payload["cameras"].setdefault("child", {})
        payload["cameras"]["parent"]["image_directory"] = str(self.session.parent_dir)
        payload["cameras"]["child"]["image_directory"] = str(self.session.child_dir)
        payload["cameras"]["parent"]["intrinsics_path"] = (
            self.parent_camera_payload.get("intrinsics_path")
        )
        payload["cameras"]["child"]["intrinsics_path"] = self.child_camera_payload.get(
            "intrinsics_path"
        )
        payload.setdefault("output", {})
        return payload

    def _summarize_result(
        self,
        result: dict[str, Any],
        *,
        output_dir: Path,
        stage: str,
    ) -> dict[str, Any]:
        summary = {
            "stage": stage,
            "success": bool(result.get("success")),
            "output_dir": str(output_dir),
            "sample_count": int(self.sample_count),
        }
        if not bool(result.get("success")):
            summary["failure_message"] = str(result.get("failure_message") or "")
            extraction_report = result.get("extraction_report") or {}
            summary["accepted_pair_count"] = int(
                extraction_report.get("pre_optimization_accepted_pair_count", 0)
            )
            summary["skip_reason_counts"] = (
                extraction_report.get("skip_reason_counts") or {}
            )
            return summary
        metrics_output = result.get("metrics_output") or {}
        metrics_summary = metrics_output.get("summary") or {}
        coarse_metrics = metrics_output.get("coarse_metrics") or {}
        summary.update(
            {
                "release_ready": bool(metrics_summary.get("release_ready")),
                "final_acceptance_status": metrics_summary.get(
                    "final_acceptance_status"
                ),
                "pair_count": int(metrics_summary.get("pair_count", 0)),
                "final_rms_px": float(metrics_summary.get("final_rms_px", 0.0)),
                "final_translation_m": metrics_summary.get("final_translation_m"),
                "final_euler_deg": metrics_summary.get("final_euler_deg"),
                "accepted_pair_ratio": float(
                    coarse_metrics.get("accepted_pair_ratio", 0.0)
                ),
                "parent_image_coverage": coarse_metrics.get("parent_image_coverage"),
                "child_image_coverage": coarse_metrics.get("child_image_coverage"),
                "pose_diversity": coarse_metrics.get("pose_diversity"),
            }
        )
        return summary

    def _run_provisional_evaluation(self) -> dict[str, Any]:
        provisional_output_dir = (
            self.session.provisional_dir / f"pairs_{self.sample_count:03d}"
        ).resolve()
        result = run_reference_calibration_from_payload(
            self._provisional_payload(),
            output_dir_override=str(provisional_output_dir),
            raise_on_failure=False,
        )
        if bool(result.get("success")) and result.get("final_transform") is not None:
            self.provisional_consensus_transform = np.asarray(
                result["final_transform"], dtype=float
            )
        self.last_provisional_result = self._summarize_result(
            result,
            output_dir=provisional_output_dir,
            stage="provisional",
        )
        self._write_capture_manifest(status="collecting")
        return result

    def _run_final_calibration(self) -> dict[str, Any]:
        result = run_reference_calibration_from_payload(
            self._provisional_payload(),
            output_dir_override=str(self.output_dir),
            raise_on_failure=False,
        )
        if bool(result.get("success")) and result.get("final_transform") is not None:
            self.final_consensus_transform = np.asarray(
                result["final_transform"], dtype=float
            )
        self.last_final_result = self._summarize_result(
            result,
            output_dir=self.output_dir,
            stage="final",
        )
        self._write_capture_manifest(
            status="complete" if bool(result.get("success")) else "final_failed"
        )
        return result

    def _accepted_image_path(self, camera_name: str) -> Path:
        directory = (
            self.session.parent_dir
            if camera_name == "parent"
            else self.session.child_dir
        )
        return directory / f"sample_{self.sample_count + 1:03d}.jpg"

    def _append_sample(
        self,
        *,
        sampling: IntrinsicSamplingState,
        detection,
        image_size_wh: tuple[int, int],
        source_path: str,
    ) -> None:
        sampling.append_sample(
            np.asarray(detection.image_points, dtype=np.float32).reshape(-1, 1, 2),
            image_size_wh,
            object_points=np.asarray(detection.object_points, dtype=np.float32).reshape(
                -1, 3
            ),
            source="live_stereo",
            source_path=source_path,
        )

    def _save_pair(
        self,
        parent_frame: np.ndarray,
        child_frame: np.ndarray,
        parent_detection,
        child_detection,
        pair_report: dict[str, Any],
    ) -> None:
        parent_path = self._accepted_image_path("parent")
        child_path = self._accepted_image_path("child")
        if not cv2.imwrite(str(parent_path), parent_frame):
            raise RuntimeError(f"Failed to save parent frame to {parent_path}")
        if not cv2.imwrite(str(child_path), child_frame):
            raise RuntimeError(f"Failed to save child frame to {child_path}")
        image_size_wh = (int(parent_frame.shape[1]), int(parent_frame.shape[0]))
        self._append_sample(
            sampling=self.parent_sampling,
            detection=parent_detection,
            image_size_wh=image_size_wh,
            source_path=str(parent_path),
        )
        child_size_wh = (int(child_frame.shape[1]), int(child_frame.shape[0]))
        self._append_sample(
            sampling=self.child_sampling,
            detection=child_detection,
            image_size_wh=child_size_wh,
            source_path=str(child_path),
        )
        relative_transform = pair_report.get("relative_transform")
        if relative_transform is not None:
            self.accepted_relative_transforms.append(
                np.asarray(relative_transform, dtype=float)
            )
        self._record_pair_review(pair_report)
        next_action = (self.last_pair_review or {}).get(
            "next_action"
        ) or "Move to a new informative stereo pose."
        self.feedback_text = f"Accepted stereo pair #{self.sample_count}. {next_action}"
        self._set_diagnostics(
            parent=(
                f"Parent pose ok, reproj="
                f"{pair_report['parent']['reprojection_rms_px']:.2f}px"
            ),
            child=(
                f"Child pose ok, reproj="
                f"{pair_report['child']['reprojection_rms_px']:.2f}px"
            ),
            stereo=(
                f"Accepted pair #{self.sample_count}; "
                f"{max(self._minimum_pair_count() - self.sample_count, 0)} more "
                "pairs until full stereo review."
            ),
        )
        print(f"[OK] Accepted stereo pair #{self.sample_count}")
        print(f"[SAVED] Parent image: {parent_path}")
        print(f"[SAVED] Child image: {child_path}")
        print(
            "[INFO] Pair review:",
            f"next_action={next_action}",
            f"parent_reproj={pair_report['parent']['reprojection_rms_px']:.3f}px",
            f"child_reproj={pair_report['child']['reprojection_rms_px']:.3f}px",
        )
        self._write_capture_manifest(status="collecting")

    def _handle_stereo_capture(
        self,
        parent_frame: np.ndarray,
        child_frame: np.ndarray,
        parent_detection,
        child_detection,
    ) -> int | None:
        if not bool(parent_detection.found) or not bool(child_detection.found):
            self.parent_sampling.reset_stability()
            self.child_sampling.reset_stability()
            if not bool(parent_detection.found) and not bool(child_detection.found):
                self.feedback_text = (
                    "Both cameras lost the board. Re-center it and reduce blur."
                )
                self._set_diagnostics(
                    parent=self._missing_target_feedback("Parent"),
                    child=self._missing_target_feedback("Child"),
                    stereo="No stereo pair yet; both views must see the same board.",
                )
            elif not bool(parent_detection.found):
                self.feedback_text = (
                    "Parent camera lost the board; recover detection before capturing."
                )
                self._set_diagnostics(
                    parent=self._missing_target_feedback("Parent"),
                    child="Child target detection is ok.",
                    stereo=(
                        "Stereo pair blocked because the parent view has no "
                        "valid target."
                    ),
                )
            else:
                self.feedback_text = (
                    "Child camera lost the board; recover detection before capturing."
                )
                self._set_diagnostics(
                    parent="Parent target detection is ok.",
                    child=self._missing_target_feedback("Child"),
                    stereo=(
                        "Stereo pair blocked because the child view has no "
                        "valid target."
                    ),
                )
            return None

        if (
            not self.parent_sampling.can_capture_now()
            or not self.child_sampling.can_capture_now()
        ):
            self.feedback_text = "Waiting for capture cooldown..."
            self._set_diagnostics(
                parent="Target detected; waiting for capture cooldown.",
                child="Target detected; waiting for capture cooldown.",
                stereo="Cooldown active to avoid near-duplicate stereo pairs.",
            )
            return None

        parent_debug = self.parent_sampling.note_detection(
            parent_detection.image_points
        )
        child_debug = self.child_sampling.note_detection(child_detection.image_points)
        self.parent_last_debug = getattr(parent_detection, "debug_info", None)
        self.child_last_debug = getattr(child_detection, "debug_info", None)

        stability_target = max(
            int(self.parent_sampling.stability_frames),
            int(self.child_sampling.stability_frames),
        )
        parent_stable = (
            int(parent_debug.get("stability_counter", 0)) >= stability_target
        )
        child_stable = int(child_debug.get("stability_counter", 0)) >= stability_target
        if not parent_stable or not child_stable:
            parent_counter = int(parent_debug.get("stability_counter", 0))
            child_counter = int(child_debug.get("stability_counter", 0))
            self.feedback_text = (
                "Hold both views steady "
                f"(parent={parent_counter}/{stability_target}, "
                f"child={child_counter}/"
                f"{stability_target})"
            )
            self._set_diagnostics(
                parent="Target detected; holding for stability gate.",
                child="Target detected; holding for stability gate.",
                stereo="Keep the board still until both cameras are stable.",
            )
            return None

        parent_size = (int(parent_frame.shape[1]), int(parent_frame.shape[0]))
        child_size = (int(child_frame.shape[1]), int(child_frame.shape[0]))
        parent_decision = self.parent_sampling.evaluate_capture_candidate(
            parent_detection.image_points,
            parent_size,
        )
        child_decision = self.child_sampling.evaluate_capture_candidate(
            child_detection.image_points,
            child_size,
        )

        if bool(parent_decision.get("accept")) and bool(child_decision.get("accept")):
            pair_report = self._evaluate_candidate_pair(
                parent_detection,
                child_detection,
                parent_size=parent_size,
                child_size=child_size,
            )
            if not bool(pair_report.get("accept")):
                self.parent_sampling.reset_stability()
                self.child_sampling.reset_stability()
                self.feedback_text = str(
                    pair_report.get("feedback") or "Pair rejected."
                )
                parent_pose = pair_report.get("parent") or {}
                child_pose = pair_report.get("child") or {}
                parent_diag = "Parent pose needs improvement."
                child_diag = "Child pose needs improvement."
                if "reprojection_rms_px" in parent_pose:
                    parent_diag = (
                        "Parent reproj="
                        f"{float(parent_pose['reprojection_rms_px']):.2f}px"
                    )
                if "reprojection_rms_px" in child_pose:
                    child_diag = (
                        "Child reproj="
                        f"{float(child_pose['reprojection_rms_px']):.2f}px"
                    )
                self._set_diagnostics(
                    parent=parent_diag,
                    child=child_diag,
                    stereo=str(pair_report.get("feedback") or "Pair rejected."),
                )
                return None

            self._save_pair(
                parent_frame,
                child_frame,
                parent_detection,
                child_detection,
                pair_report,
            )
            self.parent_sampling.reset_stability()
            self.child_sampling.reset_stability()

            if self.sample_count >= self._minimum_pair_count() and (
                self.last_provisional_result is None
                or self.sample_count % self.provisional_eval_interval == 0
            ):
                result = self._run_provisional_evaluation()
                provisional_summary = self.last_provisional_result or {}
                if bool(result.get("success")):
                    release_ready = bool(provisional_summary.get("release_ready"))
                    status = provisional_summary.get("final_acceptance_status")
                    self.feedback_text = (
                        "Stereo provisional: "
                        f"{status}, release_ready={release_ready}"
                    )
                    self._set_diagnostics(
                        parent=self.current_diagnostics["parent"],
                        child=self.current_diagnostics["child"],
                        stereo=(
                            f"Stereo provisional {status}; "
                            f"release_ready={release_ready}"
                        ),
                    )
                    print(
                        "[INFO] Provisional stereo review:",
                        f"status={status}",
                        f"release_ready={release_ready}",
                        f"sample_count={self.sample_count}",
                    )
                    if (
                        self.auto_stop_on_release_ready
                        and release_ready
                        and self.sample_count >= self.required_sample_count
                    ):
                        final_result = self._run_final_calibration()
                        if not bool(final_result.get("success")):
                            return 2
                        if self.require_release_ready and not bool(
                            (self.last_final_result or {}).get("release_ready")
                        ):
                            return 3
                        return 0
                else:
                    failure_message = provisional_summary.get("failure_message")
                    self.feedback_text = (
                        "Stereo provisional failed: "
                        f"{failure_message or 'need more consistent pairs'}"
                    )
                    self._set_diagnostics(
                        parent=self.current_diagnostics["parent"],
                        child=self.current_diagnostics["child"],
                        stereo=str(self.feedback_text),
                    )

            if self.capture_only and self.sample_count >= self.required_sample_count:
                self._write_capture_manifest(status="capture_complete")
                return 0

            if (
                not self.capture_only
                and not self.auto_stop_on_release_ready
                and self.sample_count >= self.required_sample_count
            ):
                final_result = self._run_final_calibration()
                if not bool(final_result.get("success")):
                    return 2
                if self.require_release_ready and not bool(
                    (self.last_final_result or {}).get("release_ready")
                ):
                    return 3
                return 0
            return None

        self.parent_sampling.reset_stability()
        self.child_sampling.reset_stability()
        self.feedback_text = self._pair_rejection_feedback(
            parent_decision, child_decision
        )
        self._set_diagnostics(
            parent=f"Parent guidance: {self.feedback_text}",
            child=f"Child guidance: {self.feedback_text}",
            stereo="Coverage/diversity gate rejected this pose.",
        )
        return None

    def _log_headless_progress(self) -> None:
        parent_progress = self.parent_sampling.progress_snapshot()
        child_progress = self.child_sampling.progress_snapshot()
        print(
            "[INFO] Stereo headless progress:",
            f"pairs={self.sample_count}/{self.required_sample_count}",
            f"parent_stage={parent_progress['stage']}",
            f"child_stage={child_progress['stage']}",
            (
                "parent_cov="
                f"{parent_progress['coverage_cell_count']}/"
                f"{parent_progress['coverage_target_cell_count']}"
            ),
            (
                "child_cov="
                f"{child_progress['coverage_cell_count']}/"
                f"{child_progress['coverage_target_cell_count']}"
            ),
            f"feedback={self.feedback_text}",
        )
        if self.last_provisional_result is not None:
            print(
                "[INFO] Last provisional:",
                f"success={self.last_provisional_result.get('success')}",
                f"release_ready={self.last_provisional_result.get('release_ready')}",
                f"status={self.last_provisional_result.get('final_acceptance_status')}",
            )
        if self.last_pair_review is not None:
            print(
                "[INFO] Last pair review:",
                "remaining_pairs="
                f"{self.last_pair_review.get('remaining_pairs_to_full_review')}",
                f"next_action={self.last_pair_review.get('next_action')}",
            )

    def _pose_visualization_summary(self) -> dict[str, Any] | None:
        if self.last_final_result and self.last_final_result.get("success"):
            return {
                "title": "Final stereo pose",
                "translation_m": self.last_final_result.get("final_translation_m"),
                "euler_deg": self.last_final_result.get("final_euler_deg"),
                "extra": (
                    "status="
                    f"{self.last_final_result.get('final_acceptance_status')} "
                    f"release_ready={self.last_final_result.get('release_ready')}"
                ),
            }
        if self.last_provisional_result and self.last_provisional_result.get("success"):
            return {
                "title": "Provisional stereo pose",
                "translation_m": self.last_provisional_result.get(
                    "final_translation_m"
                ),
                "euler_deg": self.last_provisional_result.get("final_euler_deg"),
                "extra": (
                    "status="
                    f"{self.last_provisional_result.get('final_acceptance_status')} "
                    f"release_ready={self.last_provisional_result.get('release_ready')}"
                ),
            }
        if self.last_pair_review and self.last_pair_review.get(
            "relative_transform_summary"
        ):
            summary = copy.deepcopy(self.last_pair_review["relative_transform_summary"])
            summary["title"] = "Last accepted pair pose"
            delta = self.last_pair_review.get("relative_transform_delta_to_consensus")
            if delta is not None:
                summary["delta_to_consensus"] = delta
            permutation = self.last_pair_review.get("selected_child_permutation")
            if permutation:
                summary["extra"] = f"child_permutation={permutation}"
            return summary
        consensus_transform = self._current_consensus_transform()
        if consensus_transform is None:
            return None
        summary = self._transform_summary(consensus_transform)
        summary["title"] = "Current stereo consensus"
        return summary

    def _read_stereo_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        parent_frame = self._read_frame(self.parent_capture, "parent")
        child_frame = self._read_frame(self.child_capture, "child")
        return parent_frame, child_frame

    def _prepare_runtime_info(
        self, parent_frame: np.ndarray, child_frame: np.ndarray
    ) -> None:
        if self.parent_runtime_info is None and self.parent_capture is not None:
            self.parent_runtime_info = build_capture_runtime_info(
                self.cfg,
                self.parent_source,
                self.parent_source_meta,
                self.parent_capture,
                parent_frame,
            )
        if self.child_runtime_info is None and self.child_capture is not None:
            self.child_runtime_info = build_capture_runtime_info(
                self.cfg,
                self.child_source,
                self.child_source_meta,
                self.child_capture,
                child_frame,
            )

    def run_live_headless(self, *, max_seconds: float = 0) -> int:
        print("[INFO] Camera2Camera headless live mode started.")
        print(f"[INFO] Parent source: {self.parent_source}")
        print(f"[INFO] Child source: {self.child_source}")
        self._write_capture_manifest(status="collecting")
        if not self._open_captures():
            return 1

        start_ts = time.time()
        try:
            while True:
                parent_frame, child_frame = self._read_stereo_frames()
                if parent_frame is None or child_frame is None:
                    self.frame_counter += 1
                    if max_seconds > 0 and (time.time() - start_ts) >= float(
                        max_seconds
                    ):
                        break
                    continue

                self._prepare_runtime_info(parent_frame, child_frame)
                _parent_gray, parent_detection = self._detect(
                    parent_frame, self.parent_detector
                )
                _child_gray, child_detection = self._detect(
                    child_frame, self.child_detector
                )
                status_code = self._handle_stereo_capture(
                    parent_frame,
                    child_frame,
                    parent_detection,
                    child_detection,
                )
                if status_code is not None:
                    return status_code

                if self.frame_counter % 30 == 0:
                    self._log_headless_progress()

                if max_seconds > 0 and (time.time() - start_ts) >= float(max_seconds):
                    break
                self.frame_counter += 1
        finally:
            self._close_captures()

        if self.capture_only:
            self._write_capture_manifest(status="capture_incomplete")
            print(
                "[ERROR] Headless stereo capture did not finish.",
                f"pairs={self.sample_count}/{self.required_sample_count}",
            )
            return 2

        if self.sample_count >= self.required_sample_count:
            final_result = self._run_final_calibration()
            if not bool(final_result.get("success")):
                return 2
            if self.require_release_ready and not bool(
                (self.last_final_result or {}).get("release_ready")
            ):
                return 3
            return 0
        print(
            "[ERROR] Headless stereo live calibration did not collect enough pairs.",
            f"pairs={self.sample_count}/{self.required_sample_count}",
        )
        return 2

    def run(self, *, max_seconds: float = 0) -> int:
        self._write_capture_manifest(status="collecting")
        if not self._open_captures():
            return 1
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        window_size = display_size(self.cfg)
        cv2.resizeWindow(self.window_name, *window_size)
        start_ts = time.time()
        try:
            while True:
                parent_frame, child_frame = self._read_stereo_frames()
                if parent_frame is None or child_frame is None:
                    if max_seconds > 0 and (time.time() - start_ts) >= float(
                        max_seconds
                    ):
                        break
                    self.frame_counter += 1
                    continue
                self._prepare_runtime_info(parent_frame, child_frame)
                _parent_gray, parent_detection = self._detect(
                    parent_frame, self.parent_detector
                )
                _child_gray, child_detection = self._detect(
                    child_frame, self.child_detector
                )
                status_code = self._handle_stereo_capture(
                    parent_frame,
                    child_frame,
                    parent_detection,
                    child_detection,
                )
                parent_panel = self._draw_live_panel(
                    "Parent",
                    parent_frame,
                    parent_detection,
                    self.parent_sampling,
                )
                child_panel = self._draw_live_panel(
                    "Child",
                    child_frame,
                    child_detection,
                    self.child_sampling,
                )
                footer_lines = [
                    self.feedback_text,
                    self.current_diagnostics.get("stereo"),
                ]
                if self.last_provisional_result is not None:
                    provisional_status = self.last_provisional_result.get(
                        "final_acceptance_status"
                    )
                    provisional_release_ready = self.last_provisional_result.get(
                        "release_ready"
                    )
                    provisional_text = (
                        "Provisional "
                        f"status={provisional_status} "
                        f"release_ready={provisional_release_ready}"
                    )
                    footer_lines.append(provisional_text)
                combined = build_stereo_comparison_canvas(
                    parent_panel,
                    child_panel,
                    footer_lines=footer_lines,
                    pose_summary=self._pose_visualization_summary(),
                )
                canvas = render_preserving_aspect_ratio(combined, window_size)
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if status_code is not None:
                    return status_code
                if max_seconds > 0 and (time.time() - start_ts) >= float(max_seconds):
                    break
                self.frame_counter += 1
        finally:
            self._close_captures()
            cv2.destroyWindow(self.window_name)

        if self.capture_only:
            self._write_capture_manifest(status="capture_incomplete")
            return 2
        if self.sample_count >= self.required_sample_count:
            final_result = self._run_final_calibration()
            if not bool(final_result.get("success")):
                return 2
            if self.require_release_ready and not bool(
                (self.last_final_result or {}).get("release_ready")
            ):
                return 3
            return 0
        return 2


def run_live_capture(
    payload: dict[str, Any],
    *,
    base_directory: str | Path | None = None,
    session_name: str | None = None,
    capture_only: bool = False,
    require_release_ready: bool | None = None,
    output_dir_override: str | None = None,
    headless_live_max_seconds: float = 0,
) -> int:
    runner = StereoLiveCapture(
        payload,
        base_directory=base_directory,
        session_name=session_name,
        capture_only=capture_only,
        require_release_ready=require_release_ready,
        output_dir_override=output_dir_override,
    )
    display_available = bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    if display_available:
        try:
            return runner.run(max_seconds=0)
        except cv2.error as exc:
            print(
                f"[WARN] GUI mode failed ({exc}). Falling back to headless live mode."
            )
    else:
        print(
            "[WARN] No DISPLAY/WAYLAND_DISPLAY detected. "
            "Falling back to headless live mode."
        )
    return runner.run_live_headless(max_seconds=headless_live_max_seconds)
