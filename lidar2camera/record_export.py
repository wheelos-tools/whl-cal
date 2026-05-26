from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import yaml

from lidar2lidar.record_adapter import Record, ensure_record_available
from lidar2lidar.record_utils import discover_record_files, pointcloud_message_to_open3d


@dataclass(frozen=True)
class MessageRef:
    topic: str
    timestamp_ns: int
    record_path: str
    frame_id: str


@dataclass(frozen=True)
class ExportPair:
    index: int
    image_ref: MessageRef
    lidar_ref: MessageRef
    sync_delta_ns: int


def _frame_id_from_message(message: object) -> str:
    frame_id = getattr(message, "frame_id", "")
    if frame_id:
        return str(frame_id)
    header = getattr(message, "header", None)
    return str(getattr(header, "frame_id", ""))


def _encoding_as_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_image_message(message: object) -> np.ndarray:
    if hasattr(message, "height") and hasattr(message, "width") and hasattr(message, "encoding"):
        height = int(getattr(message, "height"))
        width = int(getattr(message, "width"))
        step = int(getattr(message, "step"))
        encoding = _encoding_as_string(getattr(message, "encoding")).lower()
        raw = np.frombuffer(bytes(getattr(message, "data")), dtype=np.uint8)

        if encoding in {"bgr8", "rgb8"}:
            channels = 3
        elif encoding in {"mono8", "8uc1"}:
            channels = 1
        else:
            raise ValueError(f"Unsupported raw image encoding: {encoding}")

        expected_row_bytes = width * channels
        if step < expected_row_bytes:
            raise ValueError(
                f"Image step {step} is smaller than expected row bytes {expected_row_bytes}."
            )
        if raw.size < height * step:
            raise ValueError(
                f"Image payload has {raw.size} bytes but expected at least {height * step}."
            )

        row_major = raw[: height * step].reshape(height, step)
        trimmed = row_major[:, :expected_row_bytes]
        if channels == 1:
            return trimmed.reshape(height, width)
        image = trimmed.reshape(height, width, channels)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    if hasattr(message, "format") and hasattr(message, "data"):
        encoded = np.frombuffer(bytes(getattr(message, "data")), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode compressed image payload.")
        return image

    raise ValueError(f"Unsupported image message type: {type(message)!r}")


def _collect_message_refs(record_files: list[str], topic: str) -> list[MessageRef]:
    refs: list[MessageRef] = []
    for record_file in record_files:
        with Record(record_file) as record:
            for channel, message, timestamp_ns in record.read_messages(topics=[topic]):
                if channel != topic:
                    continue
                refs.append(
                    MessageRef(
                        topic=channel,
                        timestamp_ns=int(timestamp_ns),
                        record_path=record_file,
                        frame_id=_frame_id_from_message(message),
                    )
                )
    refs.sort(key=lambda item: item.timestamp_ns)
    return refs


def _pair_message_refs(
    image_refs: list[MessageRef],
    lidar_refs: list[MessageRef],
    *,
    max_delta_ns: int,
) -> list[ExportPair]:
    if not image_refs or not lidar_refs:
        return []

    lidar_times = [item.timestamp_ns for item in lidar_refs]
    candidates: list[tuple[MessageRef, MessageRef, int]] = []
    for image_ref in image_refs:
        index = bisect.bisect_left(lidar_times, image_ref.timestamp_ns)
        neighbor_indices: list[int] = []
        if index < len(lidar_refs):
            neighbor_indices.append(index)
        if index > 0:
            neighbor_indices.append(index - 1)

        best_match: tuple[MessageRef, MessageRef, int] | None = None
        for neighbor_index in neighbor_indices:
            lidar_ref = lidar_refs[neighbor_index]
            delta_ns = abs(image_ref.timestamp_ns - lidar_ref.timestamp_ns)
            if delta_ns > max_delta_ns:
                continue
            if best_match is None or delta_ns < best_match[2]:
                best_match = (image_ref, lidar_ref, delta_ns)
        if best_match is not None:
            candidates.append(best_match)

    candidates.sort(key=lambda item: (item[2], item[0].timestamp_ns, item[1].timestamp_ns))
    selected: list[tuple[MessageRef, MessageRef, int]] = []
    used_image_timestamps: set[int] = set()
    used_lidar_timestamps: set[int] = set()
    for image_ref, lidar_ref, delta_ns in candidates:
        if image_ref.timestamp_ns in used_image_timestamps:
            continue
        if lidar_ref.timestamp_ns in used_lidar_timestamps:
            continue
        selected.append((image_ref, lidar_ref, delta_ns))
        used_image_timestamps.add(image_ref.timestamp_ns)
        used_lidar_timestamps.add(lidar_ref.timestamp_ns)

    selected.sort(key=lambda item: item[0].timestamp_ns)
    return [
        ExportPair(index=index, image_ref=image_ref, lidar_ref=lidar_ref, sync_delta_ns=delta_ns)
        for index, (image_ref, lidar_ref, delta_ns) in enumerate(selected)
    ]


def _summarize_float_series(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    series = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(series)),
        "p50": float(np.percentile(series, 50)),
        "p95": float(np.percentile(series, 95)),
        "max": float(np.max(series)),
        "mean": float(np.mean(series)),
    }


def _load_camera_intrinsics(camera_calibration_yaml: str) -> tuple[list[list[float]], list[float]]:
    payload = yaml.safe_load(Path(camera_calibration_yaml).read_text(encoding="utf-8")) or {}
    matrix_payload = payload.get("camera_matrix", {})
    distortion_payload = payload.get("distortion_coefficients", {})
    camera_matrix = np.asarray(matrix_payload.get("data", matrix_payload), dtype=float).reshape(3, 3)
    distortion = np.asarray(distortion_payload.get("data", distortion_payload), dtype=float).reshape(-1)
    return camera_matrix.tolist(), distortion.tolist()


def _write_lidar2camera_config(
    *,
    config_output: Path,
    data_directory: Path,
    calibration_output_dir: Path,
    camera_calibration_yaml: str,
    checkerboard_pattern_size: tuple[int, int],
    checkerboard_square_size_m: float,
    parent_frame: str,
    child_frame: str,
) -> None:
    camera_matrix, distortion = _load_camera_intrinsics(camera_calibration_yaml)
    config_payload = {
        "camera": {
            "intrinsics": camera_matrix,
            "distortion": distortion,
        },
        "checkerboard": {
            "pattern_size": [
                int(checkerboard_pattern_size[0]),
                int(checkerboard_pattern_size[1]),
            ],
            "square_size": float(checkerboard_square_size_m),
        },
        "frames": {
            "parent": str(parent_frame),
            "child": str(child_frame),
        },
        "data_directory": str(data_directory),
        "output": {
            "directory": str(calibration_output_dir),
        },
    }
    config_output.parent.mkdir(parents=True, exist_ok=True)
    with config_output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_payload, file, sort_keys=False)


def _write_manifest(
    *,
    manifest_path: Path,
    record_files: list[str],
    image_topic: str,
    lidar_topic: str,
    image_refs: list[MessageRef],
    lidar_refs: list[MessageRef],
    exported_pairs: list[dict[str, Any]],
    sync_threshold_ms: float,
    dataset_dir: Path,
    config_output: Path | None,
) -> None:
    sync_deltas_ms = [item["sync_delta_ms"] for item in exported_pairs]
    manifest_payload = {
        "summary": {
            "record_files": record_files,
            "image_topic": image_topic,
            "lidar_topic": lidar_topic,
            "image_message_count": int(len(image_refs)),
            "lidar_message_count": int(len(lidar_refs)),
            "exported_pair_count": int(len(exported_pairs)),
            "sync_threshold_ms": float(sync_threshold_ms),
            "sync_delta_ms": _summarize_float_series(sync_deltas_ms),
            "image_frame_id": image_refs[0].frame_id if image_refs else "",
            "lidar_frame_id": lidar_refs[0].frame_id if lidar_refs else "",
        },
        "artifacts": {
            "data_directory": str(dataset_dir),
            "config": None if config_output is None else str(config_output),
        },
        "pairs": exported_pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest_payload, file, sort_keys=False)


def export_record_dataset(
    *,
    record_path: str,
    image_topic: str,
    lidar_topic: str,
    output_dir: str,
    sync_threshold_ms: float,
    frame_stride: int,
    max_pairs: int,
    image_format: str,
    camera_calibration_yaml: str | None,
    checkerboard_pattern_size: tuple[int, int] | None,
    checkerboard_square_size_m: float | None,
    parent_frame: str | None,
    child_frame: str | None,
) -> dict[str, Any]:
    ensure_record_available()
    record_files = discover_record_files(record_path)
    image_refs = _collect_message_refs(record_files, image_topic)
    lidar_refs = _collect_message_refs(record_files, lidar_topic)

    pairs = _pair_message_refs(
        image_refs,
        lidar_refs,
        max_delta_ns=int(sync_threshold_ms * 1e6),
    )
    if frame_stride > 1:
        pairs = pairs[::frame_stride]
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise RuntimeError(
            "No synchronized image/LiDAR pairs were found within the requested sync threshold."
        )

    output_root = Path(output_dir).expanduser().resolve()
    dataset_dir = output_root / "calibration_data"
    diagnostics_dir = output_root / "diagnostics"
    calibration_output_dir = output_root / "calibration"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    image_lookup = {
        (pair.image_ref.record_path, pair.image_ref.timestamp_ns): pair
        for pair in pairs
    }
    lidar_lookup = {
        (pair.lidar_ref.record_path, pair.lidar_ref.timestamp_ns): pair
        for pair in pairs
    }
    exported_pairs: list[dict[str, Any]] = []
    pair_artifacts: dict[int, dict[str, Any]] = {
        pair.index: {
            "pair_id": f"pair_{pair.index:06d}",
            "image_timestamp_ns": int(pair.image_ref.timestamp_ns),
            "lidar_timestamp_ns": int(pair.lidar_ref.timestamp_ns),
            "sync_delta_ms": float(pair.sync_delta_ns / 1e6),
        }
        for pair in pairs
    }

    for record_file in record_files:
        wanted_topics = []
        if any(key[0] == record_file for key in image_lookup):
            wanted_topics.append(image_topic)
        if any(key[0] == record_file for key in lidar_lookup):
            wanted_topics.append(lidar_topic)
        if not wanted_topics:
            continue

        with Record(record_file) as record:
            for channel, message, timestamp_ns in record.read_messages(topics=wanted_topics):
                key = (record_file, int(timestamp_ns))
                pair = None
                if channel == image_topic:
                    pair = image_lookup.get(key)
                    if pair is None:
                        continue
                    image = _decode_image_message(message)
                    image_path = dataset_dir / f"pair_{pair.index:06d}.{image_format}"
                    if not cv2.imwrite(str(image_path), image):
                        raise RuntimeError(f"Failed to write image to {image_path}.")
                    pair_artifacts[pair.index]["image_path"] = str(image_path)
                elif channel == lidar_topic:
                    pair = lidar_lookup.get(key)
                    if pair is None:
                        continue
                    cloud = pointcloud_message_to_open3d(message)
                    pcd_path = dataset_dir / f"pair_{pair.index:06d}.pcd"
                    if not o3d.io.write_point_cloud(str(pcd_path), cloud, write_ascii=True):
                        raise RuntimeError(f"Failed to write point cloud to {pcd_path}.")
                    pair_artifacts[pair.index]["pcd_path"] = str(pcd_path)

    for pair in pairs:
        artifact = pair_artifacts[pair.index]
        if "image_path" not in artifact or "pcd_path" not in artifact:
            raise RuntimeError(
                f"Pair {artifact['pair_id']} was selected but its image or PCD artifact was not written."
            )
        exported_pairs.append(artifact)

    config_output = None
    resolved_parent_frame = parent_frame or (image_refs[0].frame_id if image_refs else "camera")
    resolved_child_frame = child_frame or (lidar_refs[0].frame_id if lidar_refs else lidar_topic.split("/")[-2])
    if camera_calibration_yaml is not None:
        if checkerboard_pattern_size is None or checkerboard_square_size_m is None:
            raise ValueError(
                "Generating a lidar2camera config requires both checkerboard pattern size and checkerboard square size."
            )
        config_output = output_root / "lidar2camera_config.yaml"
        _write_lidar2camera_config(
            config_output=config_output,
            data_directory=dataset_dir,
            calibration_output_dir=calibration_output_dir,
            camera_calibration_yaml=camera_calibration_yaml,
            checkerboard_pattern_size=checkerboard_pattern_size,
            checkerboard_square_size_m=checkerboard_square_size_m,
            parent_frame=resolved_parent_frame,
            child_frame=resolved_child_frame,
        )

    manifest_path = diagnostics_dir / "record_export.yaml"
    _write_manifest(
        manifest_path=manifest_path,
        record_files=record_files,
        image_topic=image_topic,
        lidar_topic=lidar_topic,
        image_refs=image_refs,
        lidar_refs=lidar_refs,
        exported_pairs=exported_pairs,
        sync_threshold_ms=sync_threshold_ms,
        dataset_dir=dataset_dir,
        config_output=config_output,
    )
    return {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "config_output": None if config_output is None else str(config_output),
        "pair_count": int(len(exported_pairs)),
        "image_frame": image_refs[0].frame_id if image_refs else "",
        "lidar_frame": lidar_refs[0].frame_id if lidar_refs else "",
    }


def _parse_pattern_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pattern size must be provided as cols,rows.")
    cols, rows = (int(parts[0]), int(parts[1]))
    if cols <= 0 or rows <= 0:
        raise argparse.ArgumentTypeError("Pattern size values must be positive integers.")
    return cols, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export synchronized Apollo record image/PCD pairs for lidar2camera."
    )
    parser.add_argument("--record-path", required=True, help="Path to a record file or split-record directory.")
    parser.add_argument("--image-topic", required=True, help="Camera image topic.")
    parser.add_argument("--lidar-topic", required=True, help="LiDAR point cloud topic.")
    parser.add_argument("--output-dir", required=True, help="Output directory for calibration_data and diagnostics.")
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=80.0,
        help="Maximum image/LiDAR timestamp gap in milliseconds.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth synchronized pair after timestamp matching.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional cap on exported synchronized pairs. 0 keeps all pairs.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="Output image format for exported camera frames.",
    )
    parser.add_argument(
        "--camera-calibration-yaml",
        default=None,
        help="Optional camera intrinsic calibration YAML used to generate lidar2camera_config.yaml.",
    )
    parser.add_argument(
        "--checkerboard-pattern-size",
        type=_parse_pattern_size,
        default=None,
        help="Checkerboard inner-corner count written to the generated config, formatted as cols,rows.",
    )
    parser.add_argument(
        "--checkerboard-square-size",
        type=float,
        default=None,
        help="Checkerboard square size in meters written to the generated config.",
    )
    parser.add_argument(
        "--parent-frame",
        default=None,
        help="Optional parent frame override for the generated lidar2camera config.",
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Optional child frame override for the generated lidar2camera config.",
    )
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be a positive integer.")
    if args.max_pairs < 0:
        raise SystemExit("--max-pairs must be >= 0.")

    result = export_record_dataset(
        record_path=args.record_path,
        image_topic=args.image_topic,
        lidar_topic=args.lidar_topic,
        output_dir=args.output_dir,
        sync_threshold_ms=args.sync_threshold_ms,
        frame_stride=args.frame_stride,
        max_pairs=args.max_pairs,
        image_format=args.image_format,
        camera_calibration_yaml=args.camera_calibration_yaml,
        checkerboard_pattern_size=args.checkerboard_pattern_size,
        checkerboard_square_size_m=args.checkerboard_square_size,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
    )
    print(yaml.safe_dump(result, sort_keys=False).strip())


if __name__ == "__main__":
    main()