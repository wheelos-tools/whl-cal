from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lidar2camera.record_export import export_record_dataset
from lidar2lidar.extrinsic_io import load_extrinsics_file


@dataclass(frozen=True)
class PreparedReferenceConfig:
    payload: dict[str, Any]
    report: dict[str, Any]


def _resolve_optional_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _record_input_enabled(record_input: dict[str, Any] | None) -> bool:
    if not isinstance(record_input, dict):
        return False
    if bool(record_input.get("enabled", False)):
        return True
    required = (
        record_input.get("record_path"),
        record_input.get("image_topic"),
        record_input.get("lidar_topic"),
    )
    return all(bool(item) for item in required)


def _merge_frame_defaults(
    payload: dict[str, Any],
    *,
    parent_frame: str | None,
    child_frame: str | None,
) -> None:
    frames = payload.setdefault("frames", {})
    if not isinstance(frames, dict):
        raise ValueError("Config field `frames` must be a mapping when provided.")
    current_parent = str(frames.get("parent", "")).strip()
    current_child = str(frames.get("child", "")).strip()
    if parent_frame and current_parent in {"", "camera"}:
        frames["parent"] = str(parent_frame)
    if child_frame and current_child in {"", "lidar"}:
        frames["child"] = str(child_frame)


def _prepare_record_input(
    payload: dict[str, Any],
    *,
    config_dir: Path,
    output_dir_override: str | None,
) -> dict[str, Any] | None:
    record_input = payload.get("record_input")
    if not _record_input_enabled(record_input):
        return None
    if not isinstance(record_input, dict):
        raise ValueError("Config field `record_input` must be a mapping.")

    output_payload = payload.get("output", {}) or {}
    output_root = Path(
        output_dir_override
        or str(output_payload.get("directory", "calibration_output"))
    ).expanduser()
    prepared_output_dir = output_root / "prepared_record_input"

    result = export_record_dataset(
        record_path=_resolve_optional_path(config_dir, record_input.get("record_path"))
        or "",
        image_topic=str(record_input.get("image_topic", "")),
        lidar_topic=str(record_input.get("lidar_topic", "")),
        output_dir=str(prepared_output_dir),
        sync_threshold_ms=float(record_input.get("sync_threshold_ms", 80.0)),
        frame_stride=int(record_input.get("frame_stride", 1)),
        max_pairs=int(record_input.get("max_pairs", 0)),
        image_format=str(record_input.get("image_format", "png")),
        camera_calibration_yaml=None,
        checkerboard_pattern_size=None,
        checkerboard_square_size_m=None,
        parent_frame=(
            None
            if record_input.get("parent_frame") is None
            else str(record_input.get("parent_frame"))
        ),
        child_frame=(
            None
            if record_input.get("child_frame") is None
            else str(record_input.get("child_frame"))
        ),
        initial_extrinsics_path=None,
    )
    payload["data_directory"] = str(result["dataset_dir"])
    _merge_frame_defaults(
        payload,
        parent_frame=result.get("image_frame"),
        child_frame=result.get("lidar_frame"),
    )
    return {
        "source": "record_input",
        "record_path": _resolve_optional_path(
            config_dir, record_input.get("record_path")
        ),
        "image_topic": str(record_input.get("image_topic", "")),
        "lidar_topic": str(record_input.get("lidar_topic", "")),
        "prepared_output_dir": str(prepared_output_dir),
        "result": copy.deepcopy(result),
    }


def _prepare_initial_transform(
    payload: dict[str, Any],
    *,
    config_dir: Path,
) -> dict[str, Any] | None:
    initial_transform_path = _resolve_optional_path(
        config_dir,
        payload.get("initial_transform_path"),
    )
    if initial_transform_path is None:
        return None
    if "initial_transform" in payload:
        raise ValueError(
            "Config cannot define both `initial_transform` and `initial_transform_path`."
        )

    matrix, parent_frame, child_frame, stamp_ns, _ = load_extrinsics_file(
        initial_transform_path
    )
    payload["initial_transform"] = copy.deepcopy(matrix.tolist())
    _merge_frame_defaults(
        payload,
        parent_frame=parent_frame,
        child_frame=child_frame,
    )
    return {
        "source": "extrinsics_file",
        "path": initial_transform_path,
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "stamp_ns": stamp_ns,
    }


def prepare_reference_config(
    config_path: str,
    *,
    output_dir_override: str | None = None,
) -> PreparedReferenceConfig:
    resolved_config_path = Path(config_path).expanduser().resolve()
    with resolved_config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError("The lidar2camera config root must be a mapping.")

    config_dir = resolved_config_path.parent
    prepared_payload = copy.deepcopy(payload)
    report = {
        "config_path": str(resolved_config_path),
        "record_input": _prepare_record_input(
            prepared_payload,
            config_dir=config_dir,
            output_dir_override=output_dir_override,
        ),
        "initial_transform": _prepare_initial_transform(
            prepared_payload,
            config_dir=config_dir,
        ),
    }
    prepared_payload["_input_preparation"] = report
    return PreparedReferenceConfig(
        payload=prepared_payload,
        report=report,
    )
