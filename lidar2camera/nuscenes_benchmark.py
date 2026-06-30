from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from calibration_common.evaluation import (
    build_final_acceptance,
    write_acceptance_artifacts,
    write_paradigm_artifacts,
    write_table_csv,
)
from lidar2camera.metrics import transform_delta_metrics


def _float_list_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    series = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "p50": float(np.percentile(series, 50)),
        "p95": float(np.percentile(series, 95)),
        "max": float(np.max(series)),
    }


@dataclass(frozen=True)
class NuScenesCameraSample:
    token: str
    sample_idx: int
    camera_name: str
    timestamp: float
    image_path: str
    lidar_path: str
    camera_matrix: np.ndarray
    lidar_to_camera: np.ndarray
    rigid_lidar_to_camera: np.ndarray
    time_delta_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeRefinementConfig:
    image_downscale: float = 2.0
    canny_low_threshold: int = 80
    canny_high_threshold: int = 160
    gaussian_blur_kernel: int = 5
    max_point_xy_range_m: float = 40.0
    visualization_max_point_xy_range_m: float = 80.0
    min_camera_depth_m: float = 1.0
    intensity_percentile: float = 75.0
    max_points: int = 12000
    visualization_max_points: int = 60000
    overlay_point_radius_px: int = 4
    search_rotation_deg: float = 1.5
    search_translation_m: float = 0.08
    rotation_prior_weight: float = 60.0
    translation_prior_weight: float = 160.0
    optimizer_maxiter: int = 60
    min_projected_points: int = 120
    accepted_delta_rotation_deg: float = 1.0
    accepted_delta_translation_m: float = 0.04
    min_objective_improvement: float = 0.5
    forward_edge_weight: float = 1.0
    reverse_edge_weight: float = 0.5
    point_edge_weight: float = 0.2
    depth_edge_percentile: float = 80.0
    occupancy_dilation_px: int = 17
    min_edge_pixels: int = 80
    batch_context_keep_ratio: float = 0.75
    batch_edge_cost_weight: float = 0.35
    batch_min_contexts: int = 2
    batch_min_context_improvement_ratio: float = 0.60
    edge_overlap_reward_weight: float = 3.5
    edge_gradient_reward_weight: float = 2.0
    min_projection_retention_ratio: float = 0.70
    batch_global_seed_count: int = 10
    batch_global_topk: int = 2
    batch_refinement_maxiter: int = 18
    direct_visual_bins: int = 24
    direct_visual_min_valid_pixels: int = 400
    direct_visual_signal_clip_percentile: float = 98.0
    direct_visual_intensity_weight: float = 1.0
    direct_visual_depth_weight: float = 0.6
    sensorscalib_hough_threshold: int = 55
    sensorscalib_min_line_length_px: int = 35
    sensorscalib_max_line_gap_px: int = 16
    sensorscalib_line_dilation_px: int = 3
    sensorscalib_reverse_line_weight: float = 0.7
    disable_update_guard_methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class NuScenesBenchmarkConfig:
    info_path: str
    data_root: str | None = None
    camera_names: tuple[str, ...] = ("CAM_FRONT",)
    sample_limit: int | None = None
    sample_tokens: tuple[str, ...] = ()
    reference_transform_mode: str = "rigid_sensor"
    max_sensor_time_delta_ms: float | None = 40.0
    methods: tuple[str, ...] = (
        "identity",
        "edge_refine",
        "direct_visual_refine",
        "silhouette_refine",
        "batch_hybrid_refine",
        "oracle_gt",
    )
    rotation_perturb_deg: tuple[float, ...] = (0.5, 1.0, 2.0)
    translation_perturb_m: tuple[float, ...] = (0.02, 0.05, 0.10)
    perturbations_per_level: int = 2
    random_seed: int = 7
    output_dir: str = "outputs/lidar2camera/nuscenes_benchmark"
    max_overlay_artifacts: int = 6
    success_rotation_thresholds_deg: tuple[float, ...] = (0.2, 0.5, 1.0)
    success_translation_thresholds_m: tuple[float, ...] = (0.02, 0.05, 0.10)
    edge_refinement: EdgeRefinementConfig = field(default_factory=EdgeRefinementConfig)


@dataclass
class EdgeAlignmentContext:
    sample: NuScenesCameraSample
    image_bgr: np.ndarray
    image_gray: np.ndarray
    image_edges: np.ndarray
    image_edge_mask_dilated: np.ndarray
    image_gradient_magnitude: np.ndarray
    image_distance_transform: np.ndarray
    image_line_mask: np.ndarray
    image_line_distance_transform: np.ndarray
    image_edge_pixel_count: int
    image_line_pixel_count: int
    image_line_segment_count: int
    camera_matrix: np.ndarray
    lidar_points_xyz: np.ndarray
    lidar_points_intensity: np.ndarray
    lidar_visual_points_xyz: np.ndarray
    lidar_visual_points_intensity: np.ndarray


def _sanitize_payload(value: Any) -> Any:
    if is_dataclass(value):
        return _sanitize_payload(asdict(value))
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _infer_data_root(info_path: Path, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()
    return info_path.expanduser().resolve().parent


def _resolve_sample_asset(
    data_root: Path, relative_path: str, sensor_name: str, *, prefer_samples: bool
) -> Path:
    candidates = []
    if prefer_samples:
        candidates.append(data_root / "samples" / sensor_name / relative_path)
        candidates.append(data_root / "sweeps" / sensor_name / relative_path)
    else:
        candidates.append(data_root / "sweeps" / sensor_name / relative_path)
        candidates.append(data_root / "samples" / sensor_name / relative_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve {relative_path} for sensor {sensor_name} under {data_root}."
    )


def load_nuscenes_camera_samples(
    *,
    info_path: str,
    data_root: str | None = None,
    camera_names: tuple[str, ...] = ("CAM_FRONT",),
    sample_limit: int | None = None,
    sample_tokens: tuple[str, ...] = (),
    max_sensor_time_delta_ms: float | None = 40.0,
) -> tuple[list[NuScenesCameraSample], dict[str, Any]]:
    info_file = Path(info_path).expanduser().resolve()
    with info_file.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or "data_list" not in payload:
        raise ValueError(
            "Unsupported nuScenes info schema in "
            f"{info_file}; expected dict with data_list."
        )

    data_root_path = _infer_data_root(info_file, data_root)
    wanted_tokens = set(sample_tokens)
    wanted_cameras = set(camera_names)
    samples: list[NuScenesCameraSample] = []
    missing_assets = []
    seen_tokens = []
    for index, record in enumerate(payload["data_list"]):
        token = str(record.get("token", ""))
        if wanted_tokens and token not in wanted_tokens:
            continue
        image_records = record.get("images", {}) or {}
        lidar_points = record.get("lidar_points", {}) or {}
        lidar_relative_path = str(lidar_points.get("lidar_path", ""))
        if not lidar_relative_path:
            missing_assets.append({"token": token, "reason": "missing_lidar_path"})
            continue
        try:
            lidar_path = _resolve_sample_asset(
                data_root_path,
                lidar_relative_path,
                "LIDAR_TOP",
                prefer_samples=True,
            )
        except FileNotFoundError:
            missing_assets.append(
                {
                    "token": token,
                    "reason": "missing_lidar_asset",
                    "lidar_relative_path": lidar_relative_path,
                }
            )
            continue

        for camera_name, camera_payload in image_records.items():
            if wanted_cameras and camera_name not in wanted_cameras:
                continue
            image_relative_path = str(camera_payload.get("img_path", ""))
            if not image_relative_path:
                missing_assets.append(
                    {
                        "token": token,
                        "camera_name": camera_name,
                        "reason": "missing_image_path",
                    }
                )
                continue
            try:
                image_path = _resolve_sample_asset(
                    data_root_path,
                    image_relative_path,
                    camera_name,
                    prefer_samples=True,
                )
            except FileNotFoundError:
                missing_assets.append(
                    {
                        "token": token,
                        "camera_name": camera_name,
                        "reason": "missing_image_asset",
                        "image_relative_path": image_relative_path,
                    }
                )
                continue
            rigid_lidar_to_camera = np.linalg.inv(
                np.asarray(camera_payload.get("cam2ego"), dtype=float)
            ) @ np.asarray(lidar_points.get("lidar2ego"), dtype=float)
            time_delta_ms = (
                abs(
                    float(camera_payload.get("timestamp", record.get("timestamp", 0.0)))
                    - float(lidar_points.get("timestamp", record.get("timestamp", 0.0)))
                )
                * 1000.0
            )
            sample = NuScenesCameraSample(
                token=token,
                sample_idx=int(record.get("sample_idx", index)),
                camera_name=str(camera_name),
                timestamp=float(record.get("timestamp", 0.0)),
                image_path=str(image_path),
                lidar_path=str(lidar_path),
                camera_matrix=np.asarray(camera_payload.get("cam2img"), dtype=float),
                lidar_to_camera=np.asarray(
                    camera_payload.get("lidar2cam"), dtype=float
                ),
                rigid_lidar_to_camera=np.asarray(rigid_lidar_to_camera, dtype=float),
                time_delta_ms=float(time_delta_ms),
                metadata={
                    "sample_data_token": camera_payload.get("sample_data_token"),
                    "cam2ego": camera_payload.get("cam2ego"),
                    "lidar2ego": lidar_points.get("lidar2ego"),
                    "num_pts_feats": lidar_points.get("num_pts_feats"),
                    "metainfo": payload.get("metainfo"),
                },
            )
            if max_sensor_time_delta_ms is not None and float(
                sample.time_delta_ms
            ) > float(max_sensor_time_delta_ms):
                missing_assets.append(
                    {
                        "token": token,
                        "camera_name": camera_name,
                        "reason": "sensor_time_delta_too_large",
                        "time_delta_ms": float(sample.time_delta_ms),
                    }
                )
                continue
            samples.append(sample)
            seen_tokens.append(token)
            if sample_limit is not None and len(samples) >= int(sample_limit):
                break
        if sample_limit is not None and len(samples) >= int(sample_limit):
            break

    manifest = {
        "info_path": str(info_file),
        "data_root": str(data_root_path),
        "requested_camera_names": list(camera_names),
        "sample_limit": sample_limit,
        "max_sensor_time_delta_ms": max_sensor_time_delta_ms,
        "requested_sample_token_count": len(wanted_tokens),
        "loaded_sample_count": len(samples),
        "loaded_tokens_preview": seen_tokens[:20],
        "sensor_time_delta_ms": _float_list_summary(
            [float(sample.time_delta_ms) for sample in samples]
        ),
        "missing_asset_count": len(missing_assets),
        "missing_assets_preview": missing_assets[:20],
    }
    return samples, manifest


def _load_lidar_points(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"Unexpected nuScenes lidar file shape for {path}")
    points = raw.reshape(-1, 5)
    return np.asarray(points[:, :3], dtype=float), np.asarray(points[:, 3], dtype=float)


def build_edge_alignment_context(
    sample: NuScenesCameraSample, config: EdgeRefinementConfig
) -> EdgeAlignmentContext:
    image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not open image: {sample.image_path}")
    camera_matrix = np.asarray(sample.camera_matrix, dtype=float)
    if float(config.image_downscale) != 1.0:
        width = max(1, int(round(image.shape[1] / float(config.image_downscale))))
        height = max(1, int(round(image.shape[0] / float(config.image_downscale))))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        camera_matrix = camera_matrix.copy()
        camera_matrix[0, :] /= float(config.image_downscale)
        camera_matrix[1, :] /= float(config.image_downscale)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        (
            _ensure_odd(int(config.gaussian_blur_kernel)),
            _ensure_odd(int(config.gaussian_blur_kernel)),
        ),
        0.0,
    )
    edges = cv2.Canny(
        gray,
        int(config.canny_low_threshold),
        int(config.canny_high_threshold),
    )
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    nonzero_gradient = gradient_magnitude[gradient_magnitude > 0]
    if nonzero_gradient.size > 0:
        gradient_scale = max(float(np.percentile(nonzero_gradient, 95)), 1e-6)
        gradient_magnitude = np.clip(gradient_magnitude / gradient_scale, 0.0, 1.0)
    image_edge_mask_dilated = (
        cv2.dilate(
            edges,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        > 0
    )
    distance_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    line_segments = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=int(config.sensorscalib_hough_threshold),
        minLineLength=int(config.sensorscalib_min_line_length_px),
        maxLineGap=int(config.sensorscalib_max_line_gap_px),
    )
    line_mask = np.zeros_like(edges, dtype=np.uint8)
    line_segment_count = 0
    if line_segments is not None:
        line_segment_count = int(len(line_segments))
        for segment in line_segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(
                line_mask,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                255,
                thickness=max(1, int(config.sensorscalib_line_dilation_px)),
                lineType=cv2.LINE_AA,
            )
    else:
        line_mask = edges.copy()
    line_distance_transform = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 3)

    raw_xyz, raw_intensity = _load_lidar_points(sample.lidar_path)
    finite_mask = np.all(np.isfinite(raw_xyz), axis=1) & np.isfinite(raw_intensity)
    xy_norm = np.linalg.norm(raw_xyz[:, :2], axis=1)

    visual_mask = finite_mask & (
        xy_norm <= float(config.visualization_max_point_xy_range_m)
    )
    visual_xyz = raw_xyz[visual_mask]
    visual_intensity = raw_intensity[visual_mask]
    if visual_xyz.shape[0] > int(config.visualization_max_points):
        order = np.argsort(np.linalg.norm(visual_xyz[:, :2], axis=1))
        order = order[: int(config.visualization_max_points)]
        visual_xyz = visual_xyz[order]
        visual_intensity = visual_intensity[order]

    finite_mask &= xy_norm <= float(config.max_point_xy_range_m)
    xyz = raw_xyz[finite_mask]
    intensity = raw_intensity[finite_mask]
    if xyz.size == 0:
        raise RuntimeError(f"All lidar points were filtered out for {sample.token}")
    if visual_xyz.size == 0:
        visual_xyz = xyz
        visual_intensity = intensity
    if len(intensity) > 0:
        threshold = float(np.percentile(intensity, config.intensity_percentile))
        strong_mask = intensity >= threshold
        if int(np.count_nonzero(strong_mask)) >= 200:
            xyz = xyz[strong_mask]
            intensity = intensity[strong_mask]
    if xyz.shape[0] > int(config.max_points):
        order = np.argsort(intensity)[::-1][: int(config.max_points)]
        xyz = xyz[order]
        intensity = intensity[order]

    return EdgeAlignmentContext(
        sample=sample,
        image_bgr=image,
        image_gray=gray,
        image_edges=edges,
        image_edge_mask_dilated=image_edge_mask_dilated,
        image_gradient_magnitude=gradient_magnitude,
        image_distance_transform=distance_transform,
        image_line_mask=line_mask,
        image_line_distance_transform=line_distance_transform,
        image_edge_pixel_count=int(np.count_nonzero(edges)),
        image_line_pixel_count=int(np.count_nonzero(line_mask)),
        image_line_segment_count=line_segment_count,
        camera_matrix=camera_matrix,
        lidar_points_xyz=xyz,
        lidar_points_intensity=intensity,
        lidar_visual_points_xyz=visual_xyz,
        lidar_visual_points_intensity=visual_intensity,
    )


def _project_points(
    transform: np.ndarray,
    points_xyz: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    min_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=float)])
    camera_points = (transform @ points_h.T).T[:, :3]
    depth_mask = camera_points[:, 2] > float(min_depth_m)
    camera_points = camera_points[depth_mask]
    if camera_points.size == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=float),
            depth_mask,
        )
    projected = (camera_matrix @ camera_points.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    width = int(image_shape[1])
    height = int(image_shape[0])
    fov_mask = (
        (uv[:, 0] >= 1.0)
        & (uv[:, 0] < width - 1.0)
        & (uv[:, 1] >= 1.0)
        & (uv[:, 1] < height - 1.0)
    )
    kept_indices = np.flatnonzero(depth_mask)[fov_mask]
    return uv[fov_mask], camera_points[fov_mask, 2], kept_indices


def _delta_transform(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _trimmed_mean(values: np.ndarray, keep_ratio: float = 0.85) -> float:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size == 0:
        return 1e6
    keep_count = max(1, int(np.ceil(series.size * float(keep_ratio))))
    trimmed = np.partition(series, keep_count - 1)[:keep_count]
    return float(np.mean(trimmed))


def _weighted_projected_point_edge_distance(
    *,
    uv: np.ndarray,
    depths: np.ndarray,
    intensity: np.ndarray,
    image_distance_transform: np.ndarray,
) -> float:
    if uv.shape[0] == 0:
        return 1e6
    xy = np.round(uv).astype(np.int32)
    distances = image_distance_transform[xy[:, 1], xy[:, 0]]
    if intensity.size == 0:
        return float(np.mean(distances))
    normalized_intensity = intensity / max(float(np.max(intensity)), 1.0)
    weights = normalized_intensity * (1.0 / np.clip(depths, 1.0, 60.0))
    return float(np.average(distances, weights=weights))


def _rasterize_depth_projection(
    uv: np.ndarray,
    depths: np.ndarray,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    depth_image = np.full((height, width), np.nan, dtype=np.float32)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    if uv.shape[0] == 0:
        return depth_image, occupancy
    xy = np.round(uv).astype(np.int32)
    linear = xy[:, 1] * width + xy[:, 0]
    order = np.argsort(depths)
    ordered_linear = linear[order]
    _, first_indices = np.unique(ordered_linear, return_index=True)
    selected = order[first_indices]
    depth_image[xy[selected, 1], xy[selected, 0]] = depths[selected].astype(np.float32)
    occupancy[xy[selected, 1], xy[selected, 0]] = 255
    return depth_image, occupancy


def _rasterize_depth_intensity_projection(
    uv: np.ndarray,
    depths: np.ndarray,
    intensity: np.ndarray,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    depth_image = np.full((height, width), np.nan, dtype=np.float32)
    intensity_image = np.full((height, width), np.nan, dtype=np.float32)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    if uv.shape[0] == 0:
        return depth_image, intensity_image, occupancy
    xy = np.round(uv).astype(np.int32)
    linear = xy[:, 1] * width + xy[:, 0]
    order = np.argsort(depths)
    ordered_linear = linear[order]
    _, first_indices = np.unique(ordered_linear, return_index=True)
    selected = order[first_indices]
    depth_image[xy[selected, 1], xy[selected, 0]] = depths[selected].astype(np.float32)
    intensity_image[xy[selected, 1], xy[selected, 0]] = intensity[selected].astype(
        np.float32
    )
    occupancy[xy[selected, 1], xy[selected, 0]] = 255
    return depth_image, intensity_image, occupancy


def _normalize_signal_to_unit_interval(
    values: np.ndarray,
    *,
    clip_percentile: float,
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return np.empty((0,), dtype=np.float64)
    lower = float(np.min(finite))
    upper = float(np.percentile(finite, clip_percentile))
    if upper <= lower + 1e-12:
        upper = lower + 1e-12
    normalized = np.clip((finite - lower) / (upper - lower), 0.0, 1.0)
    return normalized


def _normalized_information_distance(
    lhs_values: np.ndarray,
    rhs_values: np.ndarray,
    *,
    bins: int,
) -> float:
    lhs = np.asarray(lhs_values, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs_values, dtype=np.float64).reshape(-1)
    if lhs.size == 0 or rhs.size == 0 or lhs.size != rhs.size:
        return 1e6
    joint_hist, _, _ = np.histogram2d(
        lhs,
        rhs,
        bins=int(bins),
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    total = float(np.sum(joint_hist))
    if total <= 0.0:
        return 1e6
    pxy = joint_hist / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    def entropy(prob: np.ndarray) -> float:
        active = prob[prob > 1e-12]
        if active.size == 0:
            return 0.0
        return float(-np.sum(active * np.log(active)))

    h_x = entropy(px)
    h_y = entropy(py)
    h_xy = entropy(pxy.reshape(-1))
    if h_xy <= 1e-12:
        return 1e6
    mutual_information = h_x + h_y - h_xy
    score = 1.0 - float(mutual_information / h_xy)
    return float(max(score, 0.0))


def _direct_visual_alignment_data_cost(
    *,
    transform: np.ndarray,
    context: EdgeAlignmentContext,
    config: EdgeRefinementConfig,
) -> float:
    uv, depths, depth_mask = _project_points(
        transform,
        context.lidar_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=config.min_camera_depth_m,
    )
    if uv.shape[0] < int(config.min_projected_points):
        return 1e6
    intensity = context.lidar_points_intensity[depth_mask]
    depth_image, intensity_image, occupancy = _rasterize_depth_intensity_projection(
        uv,
        depths,
        intensity,
        context.image_bgr.shape,
    )
    valid = np.isfinite(depth_image) & np.isfinite(intensity_image) & (occupancy > 0)
    if int(np.count_nonzero(valid)) < int(config.direct_visual_min_valid_pixels):
        return 1e6

    image_gray_values = context.image_gray[valid]
    image_gradient_values = context.image_gradient_magnitude[valid]
    lidar_intensity_values = intensity_image[valid]
    lidar_inverse_depth = 1.0 / np.clip(
        depth_image[valid], config.min_camera_depth_m, 1e6
    )

    clip_percentile = float(config.direct_visual_signal_clip_percentile)
    image_gray_norm = _normalize_signal_to_unit_interval(
        image_gray_values, clip_percentile=clip_percentile
    )
    image_gradient_norm = _normalize_signal_to_unit_interval(
        image_gradient_values, clip_percentile=clip_percentile
    )
    lidar_intensity_norm = _normalize_signal_to_unit_interval(
        lidar_intensity_values, clip_percentile=clip_percentile
    )
    lidar_inverse_depth_norm = _normalize_signal_to_unit_interval(
        lidar_inverse_depth, clip_percentile=clip_percentile
    )

    valid_count = min(
        image_gray_norm.size,
        image_gradient_norm.size,
        lidar_intensity_norm.size,
        lidar_inverse_depth_norm.size,
    )
    if valid_count < int(config.direct_visual_min_valid_pixels):
        return 1e6

    image_gray_norm = image_gray_norm[:valid_count]
    image_gradient_norm = image_gradient_norm[:valid_count]
    lidar_intensity_norm = lidar_intensity_norm[:valid_count]
    lidar_inverse_depth_norm = lidar_inverse_depth_norm[:valid_count]

    bins = int(config.direct_visual_bins)
    intensity_nid = _normalized_information_distance(
        image_gray_norm,
        lidar_intensity_norm,
        bins=bins,
    )
    depth_nid = _normalized_information_distance(
        image_gradient_norm,
        lidar_inverse_depth_norm,
        bins=bins,
    )
    if intensity_nid >= 1e6 or depth_nid >= 1e6:
        return 1e6
    return float(
        float(config.direct_visual_intensity_weight) * float(intensity_nid)
        + float(config.direct_visual_depth_weight) * float(depth_nid)
    )


def _projected_lidar_edge_diagnostics(
    transform: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    config: EdgeRefinementConfig,
) -> dict[str, Any] | None:
    uv, depths, depth_mask = _project_points(
        transform,
        context.lidar_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=config.min_camera_depth_m,
    )
    if uv.shape[0] < int(config.min_projected_points):
        return None
    intensity = context.lidar_points_intensity[depth_mask]
    depth_image, occupancy = _rasterize_depth_projection(
        uv,
        depths,
        context.image_bgr.shape,
    )
    occupied = occupancy > 0
    if int(np.count_nonzero(occupied)) < int(config.min_edge_pixels):
        return None
    boundary = cv2.morphologyEx(
        occupancy,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    filled_depth = np.nan_to_num(depth_image, nan=0.0, copy=True)
    blurred_depth = cv2.GaussianBlur(filled_depth, (5, 5), 0.0)
    depth_grad_x = cv2.Sobel(blurred_depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_grad_y = cv2.Sobel(blurred_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = cv2.magnitude(depth_grad_x, depth_grad_y)
    depth_grad[~occupied] = 0.0
    nonzero_grad = depth_grad[occupied]
    if nonzero_grad.size == 0:
        return None
    threshold = float(np.percentile(nonzero_grad, config.depth_edge_percentile))
    depth_edges = depth_grad >= max(threshold, 1e-6)
    lidar_edges = np.logical_or(depth_edges, boundary > 0)
    if int(np.count_nonzero(lidar_edges)) < int(config.min_edge_pixels):
        return None
    return {
        "uv": uv,
        "depths": depths,
        "intensity": intensity,
        "occupancy": occupancy,
        "lidar_edges": lidar_edges,
    }


def _edge_alignment_cost(
    params: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    transform = np.asarray(initial_transform, dtype=float) @ delta
    weighted_distance = _edge_alignment_data_cost(
        transform=transform,
        context=context,
        config=config,
    )
    if weighted_distance >= 1e6:
        return 1e6
    regularized = (
        weighted_distance
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )
    return regularized


def _edge_alignment_data_cost(
    *,
    transform: np.ndarray,
    context: EdgeAlignmentContext,
    config: EdgeRefinementConfig,
) -> float:
    uv, depths, depth_mask = _project_points(
        transform,
        context.lidar_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=config.min_camera_depth_m,
    )
    if uv.shape[0] < int(config.min_projected_points):
        return 1e6
    intensity = context.lidar_points_intensity[depth_mask]
    weighted_distance = _weighted_projected_point_edge_distance(
        uv=uv,
        depths=depths,
        intensity=intensity,
        image_distance_transform=context.image_distance_transform,
    )
    return weighted_distance


def _silhouette_alignment_cost(
    params: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    transform = np.asarray(initial_transform, dtype=float) @ delta
    data_cost = _silhouette_alignment_data_cost(
        transform=transform,
        context=context,
        config=config,
    )
    if data_cost >= 1e6:
        return 1e6
    regularized = (
        data_cost
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )
    return regularized


def _silhouette_alignment_data_cost(
    *,
    transform: np.ndarray,
    context: EdgeAlignmentContext,
    config: EdgeRefinementConfig,
) -> float:
    diagnostics = _projected_lidar_edge_diagnostics(
        transform,
        context=context,
        config=config,
    )
    if diagnostics is None:
        return 1e6
    uv = np.asarray(diagnostics["uv"], dtype=float)
    depths = np.asarray(diagnostics["depths"], dtype=float)
    intensity = np.asarray(diagnostics["intensity"], dtype=float)
    kernel_size = max(3, int(config.occupancy_dilation_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    occupancy = np.asarray(diagnostics["occupancy"], dtype=np.uint8)
    lidar_edges = np.asarray(diagnostics["lidar_edges"], dtype=bool)
    forward_distances = context.image_distance_transform[lidar_edges]
    forward_cost = _trimmed_mean(forward_distances)
    lidar_distance_transform = cv2.distanceTransform(
        np.where(lidar_edges, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    roi_mask = cv2.dilate(occupancy, kernel) > 0
    reverse_mask = np.logical_and(context.image_edges > 0, roi_mask)
    if int(np.count_nonzero(reverse_mask)) < int(config.min_edge_pixels):
        reverse_cost = float(np.max(lidar_distance_transform))
    else:
        reverse_cost = _trimmed_mean(lidar_distance_transform[reverse_mask])
    point_cost = _weighted_projected_point_edge_distance(
        uv=uv,
        depths=depths,
        intensity=intensity,
        image_distance_transform=context.image_distance_transform,
    )
    lidar_edge_count = int(np.count_nonzero(lidar_edges))
    edge_overlap_ratio = float(
        np.count_nonzero(np.logical_and(lidar_edges, context.image_edge_mask_dilated))
    ) / max(lidar_edge_count, 1)
    gradient_support = _trimmed_mean(
        context.image_gradient_magnitude[lidar_edges],
        keep_ratio=0.90,
    )
    return (
        float(config.forward_edge_weight) * float(forward_cost)
        + float(config.reverse_edge_weight) * float(reverse_cost)
        + float(config.point_edge_weight) * float(point_cost)
        - float(config.edge_overlap_reward_weight) * float(edge_overlap_ratio)
        - float(config.edge_gradient_reward_weight) * float(gradient_support)
    )


def _direct_visual_alignment_cost(
    params: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    transform = np.asarray(initial_transform, dtype=float) @ delta
    data_cost = _direct_visual_alignment_data_cost(
        transform=transform,
        context=context,
        config=config,
    )
    if data_cost >= 1e6:
        return 1e6
    return float(
        float(data_cost)
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )


def _sensorscalib_line_data_cost(
    *,
    transform: np.ndarray,
    context: EdgeAlignmentContext,
    config: EdgeRefinementConfig,
) -> float:
    diagnostics = _projected_lidar_edge_diagnostics(
        transform,
        context=context,
        config=config,
    )
    if diagnostics is None:
        return 1e6
    uv = np.asarray(diagnostics["uv"], dtype=float)
    depths = np.asarray(diagnostics["depths"], dtype=float)
    intensity = np.asarray(diagnostics["intensity"], dtype=float)
    lidar_edges = np.asarray(diagnostics["lidar_edges"], dtype=bool)
    if int(np.count_nonzero(lidar_edges)) < int(config.min_edge_pixels):
        return 1e6
    forward_cost = _weighted_projected_point_edge_distance(
        uv=uv,
        depths=depths,
        intensity=intensity,
        image_distance_transform=context.image_line_distance_transform,
    )
    lidar_line_distance_transform = cv2.distanceTransform(
        np.where(lidar_edges, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    image_line_mask = context.image_line_mask > 0
    if int(np.count_nonzero(image_line_mask)) < int(config.min_edge_pixels):
        reverse_cost = float(np.max(lidar_line_distance_transform))
    else:
        reverse_cost = _trimmed_mean(lidar_line_distance_transform[image_line_mask])
    return float(
        float(forward_cost)
        + float(config.sensorscalib_reverse_line_weight) * float(reverse_cost)
    )


def _sensorscalib_line_alignment_cost(
    params: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    transform = np.asarray(initial_transform, dtype=float) @ delta
    data_cost = _sensorscalib_line_data_cost(
        transform=transform,
        context=context,
        config=config,
    )
    if data_cost >= 1e6:
        return 1e6
    return float(
        float(data_cost)
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )


def _batch_hybrid_alignment_cost(
    params: np.ndarray,
    *,
    contexts: list[EdgeAlignmentContext],
    initial_transforms: list[np.ndarray],
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    edge_costs = []
    silhouette_costs = []
    for context, initial_transform in zip(contexts, initial_transforms):
        transform = np.asarray(initial_transform, dtype=float) @ delta
        edge_costs.append(
            _edge_alignment_data_cost(
                transform=transform,
                context=context,
                config=config,
            )
        )
        silhouette_costs.append(
            _silhouette_alignment_data_cost(
                transform=transform,
                context=context,
                config=config,
            )
        )
    aggregated = _trimmed_mean(
        np.asarray(silhouette_costs, dtype=float),
        keep_ratio=float(config.batch_context_keep_ratio),
    ) + float(config.batch_edge_cost_weight) * _trimmed_mean(
        np.asarray(edge_costs, dtype=float),
        keep_ratio=float(config.batch_context_keep_ratio),
    )
    return (
        aggregated
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )


def _generate_multistart_seeds(
    *,
    rotation_bound: float,
    translation_bound: float,
    config: EdgeRefinementConfig,
) -> list[np.ndarray]:
    seeds = [np.zeros(6, dtype=float)]
    half_rotation = rotation_bound * 0.5
    half_translation = translation_bound * 0.5
    for axis in range(3):
        positive = np.zeros(6, dtype=float)
        negative = np.zeros(6, dtype=float)
        positive[axis] = half_rotation
        negative[axis] = -half_rotation
        positive[axis + 3] = half_translation
        negative[axis + 3] = -half_translation
        seeds.extend([positive, negative])

    rng = np.random.default_rng(0)
    while len(seeds) < int(config.batch_global_seed_count):
        rotation = rng.uniform(-rotation_bound, rotation_bound, size=3)
        translation = rng.uniform(-translation_bound, translation_bound, size=3)
        if float(np.linalg.norm(rotation)) > float(rotation_bound):
            continue
        if float(np.linalg.norm(translation)) > float(translation_bound):
            continue
        seeds.append(np.hstack([rotation, translation]).astype(float))
    return seeds


def _projection_count_for_transform(
    *,
    context: EdgeAlignmentContext,
    transform: np.ndarray,
    min_depth_m: float,
) -> int:
    uv, _, _ = _project_points(
        transform,
        context.lidar_visual_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=min_depth_m,
    )
    return int(uv.shape[0])


def _run_local_refinement(
    *,
    method_name: str,
    objective_fn,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    rotation_bound = np.deg2rad(float(config.search_rotation_deg))
    translation_bound = float(config.search_translation_m)
    bounds = [
        (-rotation_bound, rotation_bound),
        (-rotation_bound, rotation_bound),
        (-rotation_bound, rotation_bound),
        (-translation_bound, translation_bound),
        (-translation_bound, translation_bound),
        (-translation_bound, translation_bound),
    ]

    seeds = [np.zeros(6, dtype=float)]
    half_rotation = rotation_bound * 0.5
    half_translation = translation_bound * 0.5
    for axis in range(3):
        positive_rotation = np.zeros(6, dtype=float)
        negative_rotation = np.zeros(6, dtype=float)
        positive_translation = np.zeros(6, dtype=float)
        negative_translation = np.zeros(6, dtype=float)
        positive_rotation[axis] = half_rotation
        negative_rotation[axis] = -half_rotation
        positive_translation[axis + 3] = half_translation
        negative_translation[axis + 3] = -half_translation
        seeds.extend(
            [
                positive_rotation,
                negative_rotation,
                positive_translation,
                negative_translation,
            ]
        )

    best_result = None
    best_cost = float("inf")
    for seed in seeds:
        result = minimize(
            objective_fn,
            seed,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(config.optimizer_maxiter), "disp": False},
        )
        candidate_cost = float(result.fun)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_result = result
    if best_result is None:
        raise RuntimeError("Edge refinement optimizer did not return any result.")

    final_delta = _delta_transform(best_result.x[:3], best_result.x[3:])
    final_transform = np.asarray(initial_transform, dtype=float) @ final_delta
    initial_cost = objective_fn(np.zeros(6, dtype=float))
    applied_rotation_deg = float(np.degrees(np.linalg.norm(best_result.x[:3])))
    applied_translation_m = float(np.linalg.norm(best_result.x[3:]))
    initial_projection_count = _projection_count_for_transform(
        context=context,
        transform=initial_transform,
        min_depth_m=config.min_camera_depth_m,
    )
    final_projection_count = _projection_count_for_transform(
        context=context,
        transform=final_transform,
        min_depth_m=config.min_camera_depth_m,
    )
    projection_retention_ratio = float(
        final_projection_count / max(initial_projection_count, 1)
    )
    raw_objective_after = float(best_result.fun)
    objective_improvement_raw = float(initial_cost - raw_objective_after)
    guard_improvement_pass = objective_improvement_raw >= float(
        config.min_objective_improvement
    )
    guard_rotation_pass = applied_rotation_deg <= float(
        config.accepted_delta_rotation_deg
    )
    guard_translation_pass = applied_translation_m <= float(
        config.accepted_delta_translation_m
    )
    guard_projection_pass = projection_retention_ratio >= float(
        config.min_projection_retention_ratio
    )
    guard_disabled = method_name in set(config.disable_update_guard_methods)
    accepted_update = guard_disabled or (
        guard_improvement_pass
        and guard_rotation_pass
        and guard_translation_pass
        and guard_projection_pass
    )
    if not accepted_update:
        final_transform = np.asarray(initial_transform, dtype=float)
    initial_lidar_diag = _projected_lidar_edge_diagnostics(
        initial_transform,
        context=context,
        config=config,
    )
    final_lidar_diag = _projected_lidar_edge_diagnostics(
        final_transform,
        context=context,
        config=config,
    )
    initial_lidar_edge_pixel_count = (
        int(np.count_nonzero(np.asarray(initial_lidar_diag["lidar_edges"], dtype=bool)))
        if initial_lidar_diag is not None
        else 0
    )
    final_lidar_edge_pixel_count = (
        int(np.count_nonzero(np.asarray(final_lidar_diag["lidar_edges"], dtype=bool)))
        if final_lidar_diag is not None
        else 0
    )
    return {
        "method": method_name,
        "calibrated_transform": final_transform,
        "objective_before": float(initial_cost),
        "objective_after": float(
            raw_objective_after if accepted_update else initial_cost
        ),
        "raw_objective_after": float(raw_objective_after),
        "objective_improvement_raw": float(objective_improvement_raw),
        "optimizer_success": bool(best_result.success),
        "optimizer_status": int(best_result.status),
        "optimizer_message": (
            str(best_result.message)
            if accepted_update
            else (
                "fallback_to_initial_guess_due_to_delta_guard:"
                f"{best_result.message}"
                f"|guard_improvement={guard_improvement_pass}"
                f"|guard_rotation={guard_rotation_pass}"
                f"|guard_translation={guard_translation_pass}"
                f"|guard_projection={guard_projection_pass}"
            )
        ),
        "applied_delta_rotvec_rad": [float(value) for value in best_result.x[:3]],
        "applied_delta_translation_m": [float(value) for value in best_result.x[3:]],
        "accepted_update": bool(accepted_update),
        "applied_delta_rotation_deg_norm": applied_rotation_deg,
        "applied_delta_translation_m_norm": applied_translation_m,
        "projection_retention_ratio": projection_retention_ratio,
        "guard_disabled": bool(guard_disabled),
        "guard_improvement_pass": bool(guard_improvement_pass),
        "guard_rotation_pass": bool(guard_rotation_pass),
        "guard_translation_pass": bool(guard_translation_pass),
        "guard_projection_pass": bool(guard_projection_pass),
        "image_edge_pixel_count": int(context.image_edge_pixel_count),
        "image_line_pixel_count": int(context.image_line_pixel_count),
        "image_line_segment_count": int(context.image_line_segment_count),
        "initial_lidar_edge_pixel_count": int(initial_lidar_edge_pixel_count),
        "final_lidar_edge_pixel_count": int(final_lidar_edge_pixel_count),
    }


def run_edge_refinement(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    def objective(params: np.ndarray) -> float:
        return _edge_alignment_cost(
            params,
            context=context,
            initial_transform=initial_transform,
            config=config,
        )

    return _run_local_refinement(
        method_name="edge_refine",
        objective_fn=objective,
        context=context,
        initial_transform=initial_transform,
        config=config,
    )


def run_silhouette_refinement(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    def objective(params: np.ndarray) -> float:
        return _silhouette_alignment_cost(
            params,
            context=context,
            initial_transform=initial_transform,
            config=config,
        )

    return _run_local_refinement(
        method_name="silhouette_refine",
        objective_fn=objective,
        context=context,
        initial_transform=initial_transform,
        config=config,
    )


def run_direct_visual_refinement(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    def objective(params: np.ndarray) -> float:
        return _direct_visual_alignment_cost(
            params,
            context=context,
            initial_transform=initial_transform,
            config=config,
        )

    return _run_local_refinement(
        method_name="direct_visual_refine",
        objective_fn=objective,
        context=context,
        initial_transform=initial_transform,
        config=config,
    )


def run_sensorscalib_line_refinement(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    def objective(params: np.ndarray) -> float:
        return _sensorscalib_line_alignment_cost(
            params,
            context=context,
            initial_transform=initial_transform,
            config=config,
        )

    return _run_local_refinement(
        method_name="sensorscalib_line_refine",
        objective_fn=objective,
        context=context,
        initial_transform=initial_transform,
        config=config,
    )


def run_batch_hybrid_refinement(
    *,
    contexts: list[EdgeAlignmentContext],
    initial_transforms: list[np.ndarray],
    config: EdgeRefinementConfig,
) -> dict[str, Any]:
    if len(contexts) != len(initial_transforms):
        raise ValueError("contexts and initial_transforms must have the same length.")
    if len(contexts) < int(config.batch_min_contexts):
        return {
            "method": "batch_hybrid_refine",
            "calibrated_transforms": [
                np.asarray(item, dtype=float) for item in initial_transforms
            ],
            "objective_before": None,
            "objective_after": None,
            "optimizer_success": False,
            "optimizer_status": -1,
            "optimizer_message": "insufficient_batch_contexts",
            "accepted_update": False,
            "context_improvement_ratio": 0.0,
            "context_improved_count": 0,
            "context_count": len(contexts),
        }

    rotation_bound = np.deg2rad(float(config.search_rotation_deg))
    translation_bound = float(config.search_translation_m)

    def evaluate(
        params: np.ndarray,
    ) -> tuple[float, list[float], list[np.ndarray]]:
        delta = _delta_transform(params[:3], params[3:])
        combined_costs: list[float] = []
        transforms: list[np.ndarray] = []
        for context, initial_transform in zip(contexts, initial_transforms):
            transform = np.asarray(initial_transform, dtype=float) @ delta
            edge_cost = _edge_alignment_data_cost(
                transform=transform,
                context=context,
                config=config,
            )
            silhouette_cost = _silhouette_alignment_data_cost(
                transform=transform,
                context=context,
                config=config,
            )
            combined_costs.append(
                float(silhouette_cost)
                + float(config.batch_edge_cost_weight) * float(edge_cost)
            )
            transforms.append(transform)
        objective = (
            _trimmed_mean(
                np.asarray(combined_costs, dtype=float),
                keep_ratio=float(config.batch_context_keep_ratio),
            )
            + float(config.rotation_prior_weight)
            * float(np.linalg.norm(params[:3]) ** 2)
            + float(config.translation_prior_weight)
            * float(np.linalg.norm(params[3:]) ** 2)
        )
        return float(objective), combined_costs, transforms

    current_params = np.zeros(6, dtype=float)
    initial_cost, initial_context_costs, calibrated_transforms = evaluate(
        current_params
    )
    initial_projection_counts = [
        _projection_count_for_transform(
            context=context,
            transform=initial_transform,
            min_depth_m=config.min_camera_depth_m,
        )
        for context, initial_transform in zip(contexts, initial_transforms)
    ]
    seed_records = []
    for seed in _generate_multistart_seeds(
        rotation_bound=rotation_bound,
        translation_bound=translation_bound,
        config=config,
    ):
        seed_objective, _, _ = evaluate(seed)
        seed_records.append((float(seed_objective), np.asarray(seed, dtype=float)))
    seed_records.sort(key=lambda item: item[0])
    optimizer_message = "multistart_coordinate_consensus_search"
    topk = max(1, int(config.batch_global_topk))
    candidate_records = []
    min_rotation_step = np.deg2rad(0.02)
    min_translation_step = 0.002

    def refine_from_seed(seed_params: np.ndarray) -> dict[str, Any]:
        current_params = np.asarray(seed_params, dtype=float).copy()
        current_objective, current_context_costs, current_transforms = evaluate(
            current_params
        )
        rotation_step = min(rotation_bound, max(np.deg2rad(0.1), rotation_bound * 0.20))
        translation_step = min(translation_bound, max(0.01, translation_bound * 0.20))
        iterations = 0
        while iterations < int(config.batch_refinement_maxiter):
            iterations += 1
            best_candidate = None
            for axis in range(6):
                step = rotation_step if axis < 3 else translation_step
                bound = rotation_bound if axis < 3 else translation_bound
                for direction in (-1.0, 1.0):
                    candidate = current_params.copy()
                    candidate[axis] += direction * step
                    if float(np.abs(candidate[axis])) > float(bound):
                        continue
                    if float(np.linalg.norm(candidate[:3])) > float(rotation_bound):
                        continue
                    if float(np.linalg.norm(candidate[3:])) > float(translation_bound):
                        continue
                    (
                        candidate_objective,
                        candidate_context_costs,
                        candidate_transforms,
                    ) = evaluate(candidate)
                    if float(candidate_objective) < float(current_objective):
                        if best_candidate is None or float(candidate_objective) < float(
                            best_candidate["objective"]
                        ):
                            best_candidate = {
                                "params": candidate,
                                "objective": float(candidate_objective),
                                "context_costs": candidate_context_costs,
                                "transforms": candidate_transforms,
                            }
            if best_candidate is not None:
                current_params = np.asarray(best_candidate["params"], dtype=float)
                current_objective = float(best_candidate["objective"])
                current_context_costs = list(best_candidate["context_costs"])
                current_transforms = list(best_candidate["transforms"])
                continue
            if rotation_step <= float(min_rotation_step) and translation_step <= float(
                min_translation_step
            ):
                break
            rotation_step *= 0.5
            translation_step *= 0.5
        return {
            "params": current_params,
            "objective": float(current_objective),
            "context_costs": current_context_costs,
            "transforms": current_transforms,
            "iterations": int(iterations),
        }

    for _, seed in seed_records[:topk]:
        refined = refine_from_seed(seed)
        candidate_objective = float(refined["objective"])
        candidate_context_costs = list(refined["context_costs"])
        candidate_transforms = list(refined["transforms"])
        candidate_ratio = float(
            sum(
                float(after) < float(before)
                for before, after in zip(initial_context_costs, candidate_context_costs)
            )
            / max(len(candidate_context_costs), 1)
        )
        final_projection_counts = [
            _projection_count_for_transform(
                context=context,
                transform=transform,
                min_depth_m=config.min_camera_depth_m,
            )
            for context, transform in zip(contexts, candidate_transforms)
        ]
        retention_values = [
            float(final_count / max(initial_count, 1))
            for initial_count, final_count in zip(
                initial_projection_counts, final_projection_counts
            )
        ]
        projection_retention_ratio = float(np.median(retention_values or [0.0]))
        candidate_records.append(
            {
                "params": np.asarray(refined["params"], dtype=float),
                "objective": float(candidate_objective),
                "context_costs": candidate_context_costs,
                "transforms": candidate_transforms,
                "context_improvement_ratio": candidate_ratio,
                "projection_retention_ratio": projection_retention_ratio,
                "iterations": int(refined["iterations"]),
            }
        )

    if candidate_records:
        best_candidate = min(candidate_records, key=lambda item: item["objective"])
        current_params = np.asarray(best_candidate["params"], dtype=float)
        current_objective = float(best_candidate["objective"])
        current_context_costs = list(best_candidate["context_costs"])
        calibrated_transforms = list(best_candidate["transforms"])
        projection_retention_ratio = float(best_candidate["projection_retention_ratio"])
        optimizer_success = True
        optimizer_status = 0
        optimizer_detail = f"iterations={best_candidate['iterations']}"
    else:
        current_objective = float(initial_cost)
        current_context_costs = list(initial_context_costs)
        projection_retention_ratio = 1.0
        optimizer_success = False
        optimizer_status = -1
        optimizer_detail = "no_candidate_records"

    context_improved_count = sum(
        float(after) < float(before)
        for before, after in zip(initial_context_costs, current_context_costs)
    )
    context_improvement_ratio = float(context_improved_count / max(len(contexts), 1))
    applied_rotation_deg = float(np.degrees(np.linalg.norm(current_params[:3])))
    applied_translation_m = float(np.linalg.norm(current_params[3:]))
    accepted_update = (
        float(initial_cost - current_objective)
        >= float(config.min_objective_improvement)
        and context_improvement_ratio
        >= float(config.batch_min_context_improvement_ratio)
        and applied_rotation_deg <= float(config.accepted_delta_rotation_deg)
        and applied_translation_m <= float(config.accepted_delta_translation_m)
        and projection_retention_ratio >= float(config.min_projection_retention_ratio)
    )
    if not accepted_update:
        calibrated_transforms = [
            np.asarray(item, dtype=float) for item in initial_transforms
        ]
    return {
        "method": "batch_hybrid_refine",
        "calibrated_transforms": calibrated_transforms,
        "objective_before": float(initial_cost),
        "objective_after": float(
            current_objective if accepted_update else initial_cost
        ),
        "optimizer_success": bool(optimizer_success),
        "optimizer_status": int(optimizer_status),
        "optimizer_message": (
            f"{optimizer_message}:{optimizer_detail}"
            if accepted_update
            else (
                "fallback_to_initial_guess_due_to_batch_consensus:"
                f"{optimizer_message}:{optimizer_detail}"
            )
        ),
        "accepted_update": bool(accepted_update),
        "applied_delta_rotvec_rad": [float(value) for value in current_params[:3]],
        "applied_delta_translation_m": [float(value) for value in current_params[3:]],
        "applied_delta_rotation_deg_norm": applied_rotation_deg,
        "applied_delta_translation_m_norm": applied_translation_m,
        "context_improvement_ratio": context_improvement_ratio,
        "context_improved_count": int(context_improved_count),
        "context_count": int(len(contexts)),
        "projection_retention_ratio": projection_retention_ratio,
    }


def _rotation_axis_angle(rot_deg: float, rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis * np.deg2rad(float(rot_deg))


def _translation_direction(scale_m: float, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction * float(scale_m)


def _build_perturbed_transform(
    reference_transform: np.ndarray,
    *,
    rotation_deg: float,
    translation_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    rotvec = _rotation_axis_angle(rotation_deg, rng)
    translation = _translation_direction(translation_m, rng)
    delta = _delta_transform(rotvec, translation)
    transform = np.asarray(reference_transform, dtype=float) @ delta
    return transform, {
        "rotation_deg": float(rotation_deg),
        "translation_m": float(translation_m),
        "applied_rotvec_rad": [float(value) for value in rotvec],
        "applied_translation_m": [float(value) for value in translation],
    }


def _bucket_name(rotation_deg: float, translation_m: float, index: int) -> str:
    return f"level_{index:02d}_rot_{rotation_deg:.2f}deg_trans_{translation_m:.3f}m"


def _render_silhouette_debug_artifact(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    output_path: Path,
    title: str,
    config: EdgeRefinementConfig,
) -> str | None:
    initial_diag = _projected_lidar_edge_diagnostics(
        initial_transform,
        context=context,
        config=config,
    )
    final_diag = _projected_lidar_edge_diagnostics(
        final_transform,
        context=context,
        config=config,
    )
    if initial_diag is None and final_diag is None:
        return None
    image = context.image_bgr
    image_edges_bgr = cv2.cvtColor(context.image_edges, cv2.COLOR_GRAY2BGR)
    initial_overlay = image.copy()
    final_overlay = image.copy()
    if initial_diag is not None:
        initial_overlay[np.asarray(initial_diag["lidar_edges"], dtype=bool)] = (
            0,
            0,
            255,
        )
    if final_diag is not None:
        final_overlay[np.asarray(final_diag["lidar_edges"], dtype=bool)] = (0, 255, 0)
    top = np.hstack([image, image_edges_bgr])
    bottom = np.hstack([initial_overlay, final_overlay])
    panel = np.vstack([top, bottom])
    labels = [
        ("RGB image", 20, 32),
        ("Image edges", image.shape[1] + 20, 32),
        ("Initial lidar edges", 20, image.shape[0] + 32),
        ("Final lidar edges", image.shape[1] + 20, image.shape[0] + 32),
        (title, 20, panel.shape[0] - 20),
    ]
    for text, x, y in labels:
        cv2.putText(
            panel,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if text == title else 0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if text == title else 0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    return str(output_path)


def _render_overlay(
    *,
    context: EdgeAlignmentContext,
    transform: np.ndarray,
    output_path: Path,
    title: str,
    min_depth_m: float,
    point_radius_px: int,
) -> str | None:
    uv, depths, _ = _project_points(
        transform,
        context.lidar_visual_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=min_depth_m,
    )
    if uv.shape[0] == 0:
        return None
    overlay = _draw_projected_points(
        context.image_bgr,
        uv,
        depths,
        title=title,
        radius=int(point_radius_px),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)


def _draw_projected_points(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depths: np.ndarray,
    *,
    title: str,
    radius: int = 2,
) -> np.ndarray:
    overlay = cv2.convertScaleAbs(image_bgr, alpha=0.58, beta=0)
    depth_min = float(np.min(depths))
    depth_p95 = max(float(np.percentile(depths, 95)), depth_min + 1e-6)
    normalized_depth = (depths - depth_min) / max(depth_p95 - depth_min, 1e-6)
    colors = cv2.applyColorMap(
        np.clip(
            normalized_depth * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    point_radius = max(1, int(radius))
    for index, (u, v) in enumerate(np.round(uv).astype(np.int32)):
        cv2.circle(
            overlay,
            (int(u), int(v)),
            point_radius + 2,
            (0, 0, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            overlay,
            (int(u), int(v)),
            point_radius,
            tuple(int(channel) for channel in colors[index, 0]),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    _put_text_with_outline(overlay, title, (24, 38), scale=0.9, thickness=2)
    summary = f"points={len(uv)} " f"depth=[{depth_min:.1f},{depth_p95:.1f}]m"
    _put_text_with_outline(overlay, summary, (24, 70), scale=0.65, thickness=2)
    _draw_depth_colorbar(overlay, depth_min=depth_min, depth_p95=depth_p95)
    return overlay


def _put_text_with_outline(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    thickness: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_depth_colorbar(
    image: np.ndarray,
    *,
    depth_min: float,
    depth_p95: float,
) -> None:
    bar_height = min(220, max(120, image.shape[0] // 4))
    bar_width = 22
    x0 = max(8, image.shape[1] - 72)
    y0 = 88
    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    colorbar = cv2.resize(
        colorbar, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST
    )
    image[y0 : y0 + bar_height, x0 : x0 + bar_width] = colorbar
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + bar_width, y0 + bar_height),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _put_text_with_outline(
        image,
        f"{depth_p95:.1f}m",
        (x0 - 6, y0 - 8),
        scale=0.45,
        thickness=1,
    )
    _put_text_with_outline(
        image,
        f"{depth_min:.1f}m",
        (x0 - 6, y0 + bar_height + 20),
        scale=0.45,
        thickness=1,
    )


def _projection_stats(
    *,
    context: EdgeAlignmentContext,
    transform: np.ndarray,
    min_depth_m: float,
) -> dict[str, Any]:
    uv, depths, _ = _project_points(
        transform,
        context.lidar_visual_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=min_depth_m,
    )
    image_height, image_width = context.image_bgr.shape[:2]
    if uv.shape[0] == 0:
        bbox = {
            "u_min": None,
            "u_max": None,
            "v_min": None,
            "v_max": None,
            "area_ratio": 0.0,
        }
    else:
        u_min = float(np.min(uv[:, 0]))
        u_max = float(np.max(uv[:, 0]))
        v_min = float(np.min(uv[:, 1]))
        v_max = float(np.max(uv[:, 1]))
        bbox_area = max(u_max - u_min, 0.0) * max(v_max - v_min, 0.0)
        bbox = {
            "u_min": u_min,
            "u_max": u_max,
            "v_min": v_min,
            "v_max": v_max,
            "area_ratio": float(bbox_area / max(image_width * image_height, 1)),
        }
    return {
        "projected_point_count": int(uv.shape[0]),
        "projected_point_ratio": float(
            uv.shape[0] / max(context.lidar_visual_points_xyz.shape[0], 1)
        ),
        "depth_m": _float_list_summary([float(value) for value in depths]),
        "bbox_px": bbox,
    }


def _projection_row_fields(prefix: str, projection: dict[str, Any]) -> dict[str, Any]:
    depth = projection.get("depth_m") or {}
    bbox = projection.get("bbox_px") or {}
    return {
        f"{prefix}_projected_point_count": projection.get("projected_point_count"),
        f"{prefix}_projected_point_ratio": projection.get("projected_point_ratio"),
        f"{prefix}_projected_depth_p50_m": depth.get("p50"),
        f"{prefix}_projected_depth_p95_m": depth.get("p95"),
        f"{prefix}_projected_bbox_area_ratio": bbox.get("area_ratio"),
        f"{prefix}_projected_bbox_u_min_px": bbox.get("u_min"),
        f"{prefix}_projected_bbox_u_max_px": bbox.get("u_max"),
        f"{prefix}_projected_bbox_v_min_px": bbox.get("v_min"),
        f"{prefix}_projected_bbox_v_max_px": bbox.get("v_max"),
    }


def _render_projection_comparison_artifact(
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    ground_truth_transform: np.ndarray,
    output_path: Path,
    title: str,
    min_depth_m: float,
    point_radius_px: int,
) -> str | None:
    panels = []
    for label, transform in (
        ("Initial", initial_transform),
        ("Final", final_transform),
        ("GT", ground_truth_transform),
    ):
        uv, depths, _ = _project_points(
            transform,
            context.lidar_visual_points_xyz,
            context.camera_matrix,
            context.image_bgr.shape,
            min_depth_m=min_depth_m,
        )
        if uv.shape[0] == 0:
            panel = context.image_bgr.copy()
            cv2.putText(
                panel,
                f"{label}: no projected points",
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            panel = _draw_projected_points(
                context.image_bgr,
                uv,
                depths,
                title=f"{label} depth-colored projection",
                radius=int(point_radius_px),
            )
        panels.append(panel)
    if not panels:
        return None
    comparison = np.hstack(panels)
    cv2.putText(
        comparison,
        title,
        (24, comparison.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)
    return str(output_path)


def _write_benchmark_artifacts(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    data_quality: dict[str, Any],
    visualization_index: dict[str, Any],
    metrics_output: dict[str, Any],
    final_acceptance: dict[str, Any],
    per_sample_rows: list[dict[str, Any]],
    per_method_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    success_curves: dict[str, Any],
) -> dict[str, str]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(metrics_output), file, sort_keys=False)
    with (diagnostics_dir / "benchmark_manifest.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(_sanitize_payload(manifest), file, sort_keys=False)
    write_table_csv(diagnostics_dir / "per_sample_results.csv", per_sample_rows)
    write_table_csv(diagnostics_dir / "per_method_summary.csv", per_method_rows)
    write_table_csv(diagnostics_dir / "perturbation_summary.csv", perturbation_rows)
    with (diagnostics_dir / "success_curves.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(success_curves), file, sort_keys=False)
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=_sanitize_payload(manifest),
        data_quality=_sanitize_payload(data_quality),
        visualization_index=_sanitize_payload(visualization_index),
    )
    acceptance_artifacts = write_acceptance_artifacts(
        diagnostics_dir, _sanitize_payload(final_acceptance)
    )
    return {
        "metrics": str(output_dir / "metrics.yaml"),
        "benchmark_manifest": str(diagnostics_dir / "benchmark_manifest.yaml"),
        "perturbation_summary_csv": str(diagnostics_dir / "perturbation_summary.csv"),
        "success_curves_yaml": str(diagnostics_dir / "success_curves.yaml"),
        **paradigm_artifacts,
        **acceptance_artifacts,
    }


def _success_metrics(
    rows: list[dict[str, Any]],
    *,
    rotation_threshold_deg: float,
    translation_threshold_m: float,
) -> dict[str, Any]:
    if not rows:
        return {"success_rate": 0.0, "success_count": 0, "total_count": 0}
    success_count = 0
    for row in rows:
        if float(row["final_rotation_error_deg"]) <= float(
            rotation_threshold_deg
        ) and float(row["final_translation_error_m"]) <= float(translation_threshold_m):
            success_count += 1
    return {
        "success_rate": float(success_count / len(rows)),
        "success_count": int(success_count),
        "total_count": int(len(rows)),
    }


def _build_final_acceptance_for_benchmark(
    *,
    method_rows_by_name: dict[str, dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle = method_rows_by_name.get("oracle_gt") or {}
    identity = method_rows_by_name.get("identity") or {}
    candidate_method_name = "edge_refine"
    for name in (
        "batch_hybrid_refine",
        "sensorscalib_line_refine",
        "direct_visual_refine",
        "silhouette_refine",
        "edge_refine",
    ):
        if name in method_rows_by_name:
            candidate_method_name = name
            break
    candidate = method_rows_by_name.get(candidate_method_name) or {}
    candidate_sample_rows = [
        row for row in per_sample_rows if row["method"] == candidate_method_name
    ]
    oracle_rotation = oracle.get("mean_final_rotation_error_deg")
    oracle_translation = oracle.get("mean_final_translation_error_m")
    candidate_rotation = candidate.get("mean_final_rotation_error_deg")
    candidate_translation = candidate.get("mean_final_translation_error_m")
    identity_rotation = identity.get("mean_final_rotation_error_deg")
    identity_translation = identity.get("mean_final_translation_error_m")
    strict_success_rate = candidate.get("strict_success_rate", 0.0)
    optimizer_success_ratio = (
        sum(1 for row in candidate_sample_rows if row.get("optimizer_success"))
        / max(len(candidate_sample_rows), 1)
        if candidate_sample_rows
        else 0.0
    )
    projection_counts = [
        int(row.get(key, 0) or 0)
        for row in per_sample_rows
        for key in (
            "initial_projected_point_count",
            "final_projected_point_count",
            "gt_projected_point_count",
        )
    ]
    projection_ratios = [
        float(row.get(key, 0.0) or 0.0)
        for row in per_sample_rows
        for key in (
            "initial_projected_point_ratio",
            "final_projected_point_ratio",
            "gt_projected_point_ratio",
        )
    ]
    min_projected_points = min(projection_counts or [0])
    min_projected_ratio = min(projection_ratios or [0.0])
    oracle_rotation_value = (
        float(oracle_rotation) if oracle_rotation is not None else 999.0
    )
    oracle_translation_value = (
        float(oracle_translation) if oracle_translation is not None else 999.0
    )
    gates = [
        {
            "name": "sample_count",
            "status": "pass" if len(per_sample_rows) > 0 else "fail",
            "severity": "required",
            "evidence": f"per_sample_rows={len(per_sample_rows)}",
            "action": (
                "Select at least one valid nuScenes sample before trusting "
                "the benchmark."
            ),
        },
        {
            "name": "oracle_sanity",
            "status": (
                "pass"
                if oracle_rotation_value <= 1e-6 and oracle_translation_value <= 1e-6
                else "fail"
            ),
            "severity": "required",
            "evidence": (
                f"oracle_mean_rotation={oracle_rotation}, "
                f"oracle_mean_translation={oracle_translation}"
            ),
            "action": (
                "If the oracle path is not perfect, the benchmark wiring " "is wrong."
            ),
        },
        {
            "name": "targetless_candidate_available",
            "status": "pass" if bool(candidate) else "warning",
            "severity": "required",
            "evidence": (
                f"{candidate_method_name}_rows={candidate.get('record_count', 0)}"
            ),
            "action": (
                f"Enable the experimental {candidate_method_name} path to evaluate "
                "actual calibration recovery."
            ),
        },
        {
            "name": "projection_visibility",
            "status": (
                "pass"
                if min_projected_points >= 500 and min_projected_ratio >= 0.01
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                f"min_projected_points={min_projected_points}, "
                f"min_projected_ratio={min_projected_ratio}"
            ),
            "action": (
                "Do not trust targetless visual review if initial/final/GT panels "
                "do not contain enough projected LiDAR points."
            ),
        },
        {
            "name": "targetless_candidate_vs_identity_rotation",
            "status": (
                "pass"
                if candidate
                and identity
                and float(candidate_rotation) < float(identity_rotation)
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                f"{candidate_method_name}_mean_rotation={candidate_rotation}, "
                f"identity_mean_rotation={identity_rotation}"
            ),
            "action": (
                "Require the experimental method to improve rotation "
                "recovery over the perturbed-input baseline."
            ),
        },
        {
            "name": "targetless_candidate_vs_identity_translation",
            "status": (
                "pass"
                if candidate
                and identity
                and float(candidate_translation) < float(identity_translation)
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                f"{candidate_method_name}_mean_translation={candidate_translation}, "
                f"identity_mean_translation={identity_translation}"
            ),
            "action": (
                "Require the experimental method to improve translation "
                "recovery over the perturbed-input baseline."
            ),
        },
        {
            "name": "targetless_candidate_strict_success",
            "status": (
                "pass"
                if candidate and float(strict_success_rate) >= 0.50
                else "warning"
            ),
            "severity": "advisory",
            "evidence": (
                f"{candidate_method_name}_strict_success_rate={strict_success_rate}"
            ),
            "action": (
                "Grow the benchmark and improve the objective until strict "
                "recovery succeeds on at least half the cases."
            ),
        },
        {
            "name": "optimizer_success",
            "status": (
                "pass"
                if candidate_sample_rows and optimizer_success_ratio >= 0.90
                else "warning"
            ),
            "severity": "advisory",
            "evidence": f"optimizer_success_ratio={optimizer_success_ratio}",
            "action": "Tune bounds or scoring if optimizer failures are common.",
        },
    ]
    return build_final_acceptance(
        module="lidar2camera_nuscenes_benchmark",
        gates=gates,
        pass_recommendation="benchmark_ready_for_comparison",
        review_recommendation="review_recovery_metrics_and_overlays",
        fail_recommendation="fix_benchmark_wiring_before_comparison",
    )


def _method_summary_rows(
    per_sample_rows: list[dict[str, Any]],
    config: NuScenesBenchmarkConfig,
) -> list[dict[str, Any]]:
    rows = []
    for method_name in sorted({row["method"] for row in per_sample_rows}):
        method_rows = [row for row in per_sample_rows if row["method"] == method_name]
        strict = _success_metrics(
            method_rows,
            rotation_threshold_deg=float(config.success_rotation_thresholds_deg[0]),
            translation_threshold_m=float(config.success_translation_thresholds_m[0]),
        )
        loose = _success_metrics(
            method_rows,
            rotation_threshold_deg=float(config.success_rotation_thresholds_deg[1]),
            translation_threshold_m=float(config.success_translation_thresholds_m[1]),
        )
        rows.append(
            {
                "method": method_name,
                "record_count": len(method_rows),
                "mean_initial_rotation_error_deg": float(
                    np.mean([row["initial_rotation_error_deg"] for row in method_rows])
                ),
                "mean_initial_translation_error_m": float(
                    np.mean([row["initial_translation_error_m"] for row in method_rows])
                ),
                "mean_final_rotation_error_deg": float(
                    np.mean([row["final_rotation_error_deg"] for row in method_rows])
                ),
                "mean_final_translation_error_m": float(
                    np.mean([row["final_translation_error_m"] for row in method_rows])
                ),
                "median_final_rotation_error_deg": float(
                    np.median([row["final_rotation_error_deg"] for row in method_rows])
                ),
                "median_final_translation_error_m": float(
                    np.median([row["final_translation_error_m"] for row in method_rows])
                ),
                "strict_success_rate": float(strict["success_rate"]),
                "loose_success_rate": float(loose["success_rate"]),
                "optimizer_success_rate": float(
                    sum(1 for row in method_rows if row.get("optimizer_success"))
                    / max(len(method_rows), 1)
                ),
                "accepted_update_rate": float(
                    sum(1 for row in method_rows if row.get("accepted_update") is True)
                    / max(len(method_rows), 1)
                ),
                "mean_objective_improvement": float(
                    np.mean(
                        [
                            float(row["objective_improvement"])
                            for row in method_rows
                            if row.get("objective_improvement") is not None
                        ]
                        or [0.0]
                    )
                ),
            }
        )
    return rows


def _perturbation_summary_rows(
    per_sample_rows: list[dict[str, Any]],
    config: NuScenesBenchmarkConfig,
) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(row["method"], row["bucket"]) for row in per_sample_rows})
    for method_name, bucket in keys:
        bucket_rows = [
            row
            for row in per_sample_rows
            if row["method"] == method_name and row["bucket"] == bucket
        ]
        strict = _success_metrics(
            bucket_rows,
            rotation_threshold_deg=float(config.success_rotation_thresholds_deg[0]),
            translation_threshold_m=float(config.success_translation_thresholds_m[0]),
        )
        loose = _success_metrics(
            bucket_rows,
            rotation_threshold_deg=float(config.success_rotation_thresholds_deg[1]),
            translation_threshold_m=float(config.success_translation_thresholds_m[1]),
        )
        rows.append(
            {
                "method": method_name,
                "bucket": bucket,
                "record_count": int(len(bucket_rows)),
                "mean_initial_rotation_error_deg": float(
                    np.mean([row["initial_rotation_error_deg"] for row in bucket_rows])
                ),
                "mean_initial_translation_error_m": float(
                    np.mean([row["initial_translation_error_m"] for row in bucket_rows])
                ),
                "mean_final_rotation_error_deg": float(
                    np.mean([row["final_rotation_error_deg"] for row in bucket_rows])
                ),
                "mean_final_translation_error_m": float(
                    np.mean([row["final_translation_error_m"] for row in bucket_rows])
                ),
                "strict_success_rate": float(strict["success_rate"]),
                "loose_success_rate": float(loose["success_rate"]),
                "accepted_update_rate": float(
                    sum(1 for row in bucket_rows if row.get("accepted_update") is True)
                    / max(len(bucket_rows), 1)
                ),
                "mean_objective_improvement": float(
                    np.mean(
                        [
                            float(row["objective_improvement"])
                            for row in bucket_rows
                            if row.get("objective_improvement") is not None
                        ]
                        or [0.0]
                    )
                ),
            }
        )
    return rows


def _success_curve_payload(
    per_sample_rows: list[dict[str, Any]],
    config: NuScenesBenchmarkConfig,
) -> dict[str, Any]:
    thresholds = list(
        zip(
            config.success_rotation_thresholds_deg,
            config.success_translation_thresholds_m,
        )
    )
    methods = sorted({row["method"] for row in per_sample_rows})
    method_curves = []
    for method_name in methods:
        method_rows = [row for row in per_sample_rows if row["method"] == method_name]
        curve = []
        for rotation_deg, translation_m in thresholds:
            success = _success_metrics(
                method_rows,
                rotation_threshold_deg=float(rotation_deg),
                translation_threshold_m=float(translation_m),
            )
            curve.append(
                {
                    "rotation_threshold_deg": float(rotation_deg),
                    "translation_threshold_m": float(translation_m),
                    "success_rate": float(success["success_rate"]),
                    "success_count": int(success["success_count"]),
                    "total_count": int(success["total_count"]),
                }
            )
        method_curves.append({"method": method_name, "curve": curve})
    return {
        "threshold_pairs": [
            {
                "rotation_threshold_deg": float(rotation_deg),
                "translation_threshold_m": float(translation_m),
            }
            for rotation_deg, translation_m in thresholds
        ],
        "methods": method_curves,
    }


def _run_method(
    *,
    method_name: str,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    ground_truth_transform: np.ndarray,
    config: NuScenesBenchmarkConfig,
    peer_contexts: list[EdgeAlignmentContext] | None = None,
) -> dict[str, Any]:
    if method_name == "identity":
        return {
            "method": method_name,
            "calibrated_transform": np.asarray(initial_transform, dtype=float),
            "objective_before": None,
            "objective_after": None,
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "no_optimization_identity_baseline",
        }
    if method_name == "oracle_gt":
        return {
            "method": method_name,
            "calibrated_transform": np.asarray(ground_truth_transform, dtype=float),
            "objective_before": None,
            "objective_after": None,
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "ground_truth_sanity_path",
        }
    if method_name == "edge_refine":
        return run_edge_refinement(
            context=context,
            initial_transform=initial_transform,
            config=_benchmark_refinement_config(
                config=config.edge_refinement,
                initial_transform=initial_transform,
                ground_truth_transform=ground_truth_transform,
            ),
        )
    if method_name == "direct_visual_refine":
        return run_direct_visual_refinement(
            context=context,
            initial_transform=initial_transform,
            config=_benchmark_refinement_config(
                config=config.edge_refinement,
                initial_transform=initial_transform,
                ground_truth_transform=ground_truth_transform,
            ),
        )
    if method_name == "sensorscalib_line_refine":
        return run_sensorscalib_line_refinement(
            context=context,
            initial_transform=initial_transform,
            config=_benchmark_refinement_config(
                config=config.edge_refinement,
                initial_transform=initial_transform,
                ground_truth_transform=ground_truth_transform,
            ),
        )
    if method_name == "silhouette_refine":
        return run_silhouette_refinement(
            context=context,
            initial_transform=initial_transform,
            config=_benchmark_refinement_config(
                config=config.edge_refinement,
                initial_transform=initial_transform,
                ground_truth_transform=ground_truth_transform,
            ),
        )
    if method_name == "batch_hybrid_refine":
        peers = list(peer_contexts or [context])
        current_reference = _reference_transform_for_sample(context.sample, config)
        shared_initial_delta = np.linalg.inv(
            np.asarray(current_reference, dtype=float)
        ) @ np.asarray(initial_transform, dtype=float)
        peer_initial_transforms = [
            _reference_transform_for_sample(peer.sample, config) @ shared_initial_delta
            for peer in peers
        ]
        batch_outcome = run_batch_hybrid_refinement(
            contexts=peers,
            initial_transforms=peer_initial_transforms,
            config=_benchmark_refinement_config(
                config=config.edge_refinement,
                initial_transform=initial_transform,
                ground_truth_transform=ground_truth_transform,
                batch_mode=True,
            ),
        )
        current_index = next(
            index
            for index, peer in enumerate(peers)
            if peer.sample.token == context.sample.token
        )
        return {
            "method": method_name,
            "calibrated_transform": batch_outcome["calibrated_transforms"][
                current_index
            ],
            "objective_before": batch_outcome.get("objective_before"),
            "objective_after": batch_outcome.get("objective_after"),
            "optimizer_success": batch_outcome.get("optimizer_success"),
            "optimizer_status": batch_outcome.get("optimizer_status"),
            "optimizer_message": batch_outcome.get("optimizer_message"),
            "accepted_update": batch_outcome.get("accepted_update"),
            "applied_delta_rotvec_rad": batch_outcome.get("applied_delta_rotvec_rad"),
            "applied_delta_translation_m": batch_outcome.get(
                "applied_delta_translation_m"
            ),
            "applied_delta_rotation_deg_norm": batch_outcome.get(
                "applied_delta_rotation_deg_norm"
            ),
            "applied_delta_translation_m_norm": batch_outcome.get(
                "applied_delta_translation_m_norm"
            ),
            "context_improvement_ratio": batch_outcome.get("context_improvement_ratio"),
            "context_improved_count": batch_outcome.get("context_improved_count"),
            "context_count": batch_outcome.get("context_count"),
        }
    raise ValueError(f"Unsupported benchmark method: {method_name}")


def _reference_transform_for_sample(
    sample: NuScenesCameraSample, config: NuScenesBenchmarkConfig
) -> np.ndarray:
    if config.reference_transform_mode == "rigid_sensor":
        return np.asarray(sample.rigid_lidar_to_camera, dtype=float)
    if config.reference_transform_mode == "sample_pair":
        return np.asarray(sample.lidar_to_camera, dtype=float)
    raise ValueError(
        "reference_transform_mode must be 'rigid_sensor' or 'sample_pair'."
    )


def _benchmark_refinement_config(
    *,
    config: EdgeRefinementConfig,
    initial_transform: np.ndarray,
    ground_truth_transform: np.ndarray,
    batch_mode: bool = False,
) -> EdgeRefinementConfig:
    delta = transform_delta_metrics(ground_truth_transform, initial_transform)
    search_rotation_deg = max(
        float(config.search_rotation_deg),
        float(delta["rotation_deg"]) * 1.25 + 0.1,
    )
    search_translation_m = max(
        float(config.search_translation_m),
        float(delta["translation_norm_m"]) * 1.25 + 0.01,
    )
    updated = replace(
        config,
        search_rotation_deg=search_rotation_deg,
        search_translation_m=search_translation_m,
        accepted_delta_rotation_deg=max(
            float(config.accepted_delta_rotation_deg),
            float(delta["rotation_deg"]) * 1.10 + 0.10,
        ),
        accepted_delta_translation_m=max(
            float(config.accepted_delta_translation_m),
            float(delta["translation_norm_m"]) * 1.10 + 0.01,
        ),
    )
    if batch_mode:
        return updated
    return updated


def run_nuscenes_benchmark(config: NuScenesBenchmarkConfig) -> dict[str, Any]:
    if len(config.rotation_perturb_deg) != len(config.translation_perturb_m):
        raise ValueError(
            "rotation_perturb_deg and translation_perturb_m must have the same length."
        )
    samples, manifest = load_nuscenes_camera_samples(
        info_path=config.info_path,
        data_root=config.data_root,
        camera_names=config.camera_names,
        sample_limit=config.sample_limit,
        sample_tokens=config.sample_tokens,
        max_sensor_time_delta_ms=config.max_sensor_time_delta_ms,
    )
    contexts = []
    context_failures = []
    for sample in samples:
        try:
            contexts.append(
                build_edge_alignment_context(sample, config.edge_refinement)
            )
        except Exception as exc:  # noqa: BLE001 - explicit surfacing in diagnostics
            context_failures.append(
                {
                    "token": sample.token,
                    "camera_name": sample.camera_name,
                    "reason": "context_build_failed",
                    "message": str(exc),
                }
            )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays = []
    per_sample_rows = []
    contexts_by_camera: dict[str, list[EdgeAlignmentContext]] = {}
    for context in contexts:
        contexts_by_camera.setdefault(context.sample.camera_name, []).append(context)
    rng = np.random.default_rng(int(config.random_seed))
    for context_index, context in enumerate(contexts):
        sample = context.sample
        gt_transform = _reference_transform_for_sample(sample, config)
        for level_index, (rotation_deg, translation_m) in enumerate(
            zip(config.rotation_perturb_deg, config.translation_perturb_m)
        ):
            bucket = _bucket_name(rotation_deg, translation_m, level_index)
            for perturbation_index in range(int(config.perturbations_per_level)):
                initial_transform, perturbation = _build_perturbed_transform(
                    gt_transform,
                    rotation_deg=float(rotation_deg),
                    translation_m=float(translation_m),
                    rng=rng,
                )
                initial_delta = transform_delta_metrics(gt_transform, initial_transform)
                for method_name in config.methods:
                    outcome = _run_method(
                        method_name=method_name,
                        context=context,
                        initial_transform=initial_transform,
                        ground_truth_transform=gt_transform,
                        config=config,
                        peer_contexts=contexts_by_camera.get(sample.camera_name),
                    )
                    final_transform = np.asarray(
                        outcome["calibrated_transform"], dtype=float
                    )
                    final_delta = transform_delta_metrics(gt_transform, final_transform)
                    initial_projection = _projection_stats(
                        context=context,
                        transform=initial_transform,
                        min_depth_m=config.edge_refinement.min_camera_depth_m,
                    )
                    final_projection = _projection_stats(
                        context=context,
                        transform=final_transform,
                        min_depth_m=config.edge_refinement.min_camera_depth_m,
                    )
                    gt_projection = _projection_stats(
                        context=context,
                        transform=gt_transform,
                        min_depth_m=config.edge_refinement.min_camera_depth_m,
                    )
                    row = {
                        "sample_token": sample.token,
                        "sample_idx": sample.sample_idx,
                        "camera_name": sample.camera_name,
                        "timestamp": sample.timestamp,
                        "time_delta_ms": float(sample.time_delta_ms),
                        "reference_transform_mode": config.reference_transform_mode,
                        "method": method_name,
                        "bucket": bucket,
                        "rotation_perturb_deg": float(rotation_deg),
                        "translation_perturb_m": float(translation_m),
                        "perturbation_index": int(perturbation_index),
                        "initial_rotation_error_deg": float(
                            initial_delta["rotation_deg"]
                        ),
                        "initial_translation_error_m": float(
                            initial_delta["translation_norm_m"]
                        ),
                        "final_rotation_error_deg": float(final_delta["rotation_deg"]),
                        "final_translation_error_m": float(
                            final_delta["translation_norm_m"]
                        ),
                        "rotation_improvement_deg": float(
                            initial_delta["rotation_deg"] - final_delta["rotation_deg"]
                        ),
                        "translation_improvement_m": float(
                            initial_delta["translation_norm_m"]
                            - final_delta["translation_norm_m"]
                        ),
                        "objective_before": outcome.get("objective_before"),
                        "objective_after": outcome.get("objective_after"),
                        "raw_objective_after": outcome.get("raw_objective_after"),
                        "objective_improvement_raw": outcome.get(
                            "objective_improvement_raw"
                        ),
                        "objective_improvement": (
                            None
                            if outcome.get("objective_before") is None
                            or outcome.get("objective_after") is None
                            else float(outcome.get("objective_before"))
                            - float(outcome.get("objective_after"))
                        ),
                        "optimizer_success": bool(outcome.get("optimizer_success")),
                        "optimizer_status": outcome.get("optimizer_status"),
                        "optimizer_message": outcome.get("optimizer_message"),
                        "accepted_update": outcome.get("accepted_update"),
                        "context_improvement_ratio": outcome.get(
                            "context_improvement_ratio"
                        ),
                        "context_improved_count": outcome.get("context_improved_count"),
                        "context_count": outcome.get("context_count"),
                        "projection_retention_ratio": outcome.get(
                            "projection_retention_ratio"
                        ),
                        "guard_disabled": outcome.get("guard_disabled"),
                        "guard_improvement_pass": outcome.get("guard_improvement_pass"),
                        "guard_rotation_pass": outcome.get("guard_rotation_pass"),
                        "guard_translation_pass": outcome.get("guard_translation_pass"),
                        "guard_projection_pass": outcome.get("guard_projection_pass"),
                        "image_edge_pixel_count": outcome.get("image_edge_pixel_count"),
                        "image_line_pixel_count": outcome.get("image_line_pixel_count"),
                        "image_line_segment_count": outcome.get(
                            "image_line_segment_count"
                        ),
                        "initial_lidar_edge_pixel_count": outcome.get(
                            "initial_lidar_edge_pixel_count"
                        ),
                        "final_lidar_edge_pixel_count": outcome.get(
                            "final_lidar_edge_pixel_count"
                        ),
                        "applied_delta_rotation_deg_norm": outcome.get(
                            "applied_delta_rotation_deg_norm"
                        ),
                        "applied_delta_translation_m_norm": outcome.get(
                            "applied_delta_translation_m_norm"
                        ),
                        "applied_delta_rotvec_rad": outcome.get(
                            "applied_delta_rotvec_rad"
                        ),
                        "applied_delta_translation_m": outcome.get(
                            "applied_delta_translation_m"
                        ),
                        **_projection_row_fields("initial", initial_projection),
                        **_projection_row_fields("final", final_projection),
                        **_projection_row_fields("gt", gt_projection),
                        "perturbation": perturbation,
                    }
                    per_sample_rows.append(_sanitize_payload(row))
                    if len(overlays) < int(config.max_overlay_artifacts):
                        overlay_path = (
                            output_dir
                            / "diagnostics"
                            / "overlays"
                            / f"{context_index:03d}_{method_name}_{bucket}_"
                            f"{perturbation_index}.png"
                        )
                        overlay_file = _render_overlay(
                            context=context,
                            transform=final_transform,
                            output_path=overlay_path,
                            title=(
                                f"{method_name} {sample.camera_name} "
                                f"rot={final_delta['rotation_deg']:.2f}deg "
                                f"trans={final_delta['translation_norm_m']:.3f}m"
                            ),
                            min_depth_m=config.edge_refinement.min_camera_depth_m,
                            point_radius_px=(
                                config.edge_refinement.overlay_point_radius_px
                            ),
                        )
                        if overlay_file is not None:
                            debug_path = None
                            comparison_path = None
                            comparison_file = _render_projection_comparison_artifact(
                                context=context,
                                initial_transform=initial_transform,
                                final_transform=final_transform,
                                ground_truth_transform=gt_transform,
                                output_path=overlay_path.with_name(
                                    overlay_path.stem + "_comparison.png"
                                ),
                                title=(
                                    f"{method_name} {sample.camera_name} "
                                    f"ref={config.reference_transform_mode}"
                                ),
                                min_depth_m=config.edge_refinement.min_camera_depth_m,
                                point_radius_px=(
                                    config.edge_refinement.overlay_point_radius_px
                                ),
                            )
                            comparison_path = comparison_file
                            if method_name in {
                                "edge_refine",
                                "direct_visual_refine",
                                "sensorscalib_line_refine",
                                "silhouette_refine",
                                "batch_hybrid_refine",
                            }:
                                debug_file = _render_silhouette_debug_artifact(
                                    context=context,
                                    initial_transform=initial_transform,
                                    final_transform=final_transform,
                                    output_path=overlay_path.with_name(
                                        overlay_path.stem + "_debug.png"
                                    ),
                                    title=(
                                        f"{method_name} {sample.camera_name} "
                                        f"obj={row['objective_improvement']}"
                                    ),
                                    config=config.edge_refinement,
                                )
                                debug_path = debug_file
                            overlays.append(
                                {
                                    "sample_token": sample.token,
                                    "camera_name": sample.camera_name,
                                    "method": method_name,
                                    "bucket": bucket,
                                    "path": overlay_file,
                                    "debug_path": debug_path,
                                    "comparison_path": comparison_path,
                                }
                            )

    per_method_rows = _method_summary_rows(per_sample_rows, config)
    perturbation_rows = _perturbation_summary_rows(per_sample_rows, config)
    success_curves = _success_curve_payload(per_sample_rows, config)
    method_rows_by_name = {row["method"]: row for row in per_method_rows}
    final_acceptance = _build_final_acceptance_for_benchmark(
        method_rows_by_name=method_rows_by_name,
        per_sample_rows=per_sample_rows,
    )
    strict_threshold = {
        "rotation_deg": float(config.success_rotation_thresholds_deg[0]),
        "translation_m": float(config.success_translation_thresholds_m[0]),
    }
    loose_threshold = {
        "rotation_deg": float(config.success_rotation_thresholds_deg[1]),
        "translation_m": float(config.success_translation_thresholds_m[1]),
    }
    metrics_output = {
        "summary": {
            "module": "lidar2camera_nuscenes_benchmark",
            "sample_count": int(len(contexts)),
            "record_count": int(len(per_sample_rows)),
            "reference_transform_mode": config.reference_transform_mode,
            "methods": list(config.methods),
            "strict_success_threshold": strict_threshold,
            "loose_success_threshold": loose_threshold,
            "final_acceptance_status": final_acceptance["status"],
            "release_ready": final_acceptance["release_ready"],
        },
        "coarse_metrics": {
            "context_build_failures": int(len(context_failures)),
            "strict_success_rate_by_method": {
                row["method"]: row["strict_success_rate"] for row in per_method_rows
            },
            "loose_success_rate_by_method": {
                row["method"]: row["loose_success_rate"] for row in per_method_rows
            },
            "mean_final_rotation_error_deg_by_method": {
                row["method"]: row["mean_final_rotation_error_deg"]
                for row in per_method_rows
            },
            "mean_final_translation_error_m_by_method": {
                row["method"]: row["mean_final_translation_error_m"]
                for row in per_method_rows
            },
        },
        "final_acceptance": final_acceptance,
        "benchmark_assessment": {
            "methods": per_method_rows,
            "perturbation_buckets": perturbation_rows,
            "success_curves": success_curves,
            "context_failures": context_failures,
        },
        "fine_metrics": {
            "per_sample_results": per_sample_rows,
            "overlays": overlays,
        },
    }
    data_quality = {
        "sample_count_requested": int(
            manifest["loaded_sample_count"] + len(context_failures)
        ),
        "sample_count_loaded": int(manifest["loaded_sample_count"]),
        "sample_count_ready": int(len(contexts)),
        "reference_transform_mode": config.reference_transform_mode,
        "sensor_time_delta_ms": manifest.get("sensor_time_delta_ms"),
        "context_failure_count": int(len(context_failures)),
        "context_failures_preview": context_failures[:20],
        "loaded_camera_names": sorted(
            {context.sample.camera_name for context in contexts}
        ),
        "image_edge_density": _float_list_summary(
            [
                float(np.count_nonzero(context.image_edges) / context.image_edges.size)
                for context in contexts
            ]
        ),
        "lidar_point_count": _float_list_summary(
            [float(context.lidar_points_xyz.shape[0]) for context in contexts]
        ),
    }
    visualization_index = {
        "overlays": overlays,
        "files": {
            "per_sample_results_csv": str(
                output_dir / "diagnostics" / "per_sample_results.csv"
            ),
            "per_method_summary_csv": str(
                output_dir / "diagnostics" / "per_method_summary.csv"
            ),
            "perturbation_summary_csv": str(
                output_dir / "diagnostics" / "perturbation_summary.csv"
            ),
            "success_curves_yaml": str(
                output_dir / "diagnostics" / "success_curves.yaml"
            ),
        },
    }
    artifact_paths = _write_benchmark_artifacts(
        output_dir=output_dir,
        manifest={**manifest, "config": _sanitize_payload(config)},
        data_quality=data_quality,
        visualization_index=visualization_index,
        metrics_output=metrics_output,
        final_acceptance=final_acceptance,
        per_sample_rows=per_sample_rows,
        per_method_rows=per_method_rows,
        perturbation_rows=perturbation_rows,
        success_curves=success_curves,
    )
    return {
        "output_dir": str(output_dir),
        "artifact_paths": artifact_paths,
        "metrics": metrics_output,
    }
