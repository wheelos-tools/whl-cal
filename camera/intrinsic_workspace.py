#!/usr/bin/env python3

"""Path helpers for camera intrinsic capture and calibration workflows."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


def _slugify(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "session"


@dataclass(frozen=True)
class CaptureSessionPaths:
    label: str
    session_dir: Path
    accepted_dir: Path
    debug_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class RunSessionPaths:
    label: str
    session_dir: Path
    calibration_yaml_path: Path
    comparison_view_path: Path


class IntrinsicWorkspace:
    def __init__(self, cfg, target_type, session_name=None):
        workflow_cfg = cfg.get("workflow", {}) or {}
        self.root_dir = Path(workflow_cfg.get("root_dir", "outputs/camera_intrinsic"))
        self.target_type = _slugify(target_type or "target")
        self.base_label = _slugify(
            session_name
            or workflow_cfg.get("session_name")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.save_live_accepted_frames = bool(
            workflow_cfg.get("save_live_accepted_frames", True)
        )

    def _with_target_suffix(self, label):
        slug = _slugify(label)
        suffix = f"_{self.target_type}"
        if slug == self.target_type or slug.endswith(suffix):
            return slug
        return f"{slug}{suffix}"

    def dataset_label_from_images_dir(self, images_dir):
        path = Path(images_dir)
        label = path.name or "images"
        if label == "accepted" and path.parent.name:
            label = path.parent.name
        return self._with_target_suffix(label)

    def prepare_capture_session(self):
        label = self._with_target_suffix(self.base_label)
        session_dir = self.root_dir / "captures" / label
        accepted_dir = session_dir / "accepted"
        debug_dir = session_dir / "debug"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        return CaptureSessionPaths(
            label=label,
            session_dir=session_dir,
            accepted_dir=accepted_dir,
            debug_dir=debug_dir,
            manifest_path=session_dir / "capture_session.yaml",
        )

    def prepare_run_session(self, dataset_label=None):
        resolved_label = self._with_target_suffix(dataset_label or self.base_label)
        run_label = f"{datetime.now():%Y%m%d_%H%M%S}_{resolved_label}"
        session_dir = self.root_dir / "runs" / run_label
        session_dir.mkdir(parents=True, exist_ok=True)
        return RunSessionPaths(
            label=run_label,
            session_dir=session_dir,
            calibration_yaml_path=session_dir / "calibration.yaml",
            comparison_view_path=session_dir / "comparison_view.png",
        )

    def accepted_sample_path(self, capture_session, sample_index):
        next_index = max(1, int(sample_index))
        while True:
            candidate = capture_session.accepted_dir / f"sample_{next_index:03d}.jpg"
            if not candidate.exists():
                return candidate
            next_index += 1

    def debug_image_path(self, capture_session, name):
        return capture_session.debug_dir / str(name)
