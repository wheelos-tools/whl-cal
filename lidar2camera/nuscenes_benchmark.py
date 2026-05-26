from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeRefinementConfig:
    image_downscale: float = 1.0
    canny_low_threshold: int = 80
    canny_high_threshold: int = 160
    gaussian_blur_kernel: int = 5
    max_point_xy_range_m: float = 40.0
    min_camera_depth_m: float = 1.0
    intensity_percentile: float = 75.0
    max_points: int = 12000
    search_rotation_deg: float = 1.5
    search_translation_m: float = 0.08
    rotation_prior_weight: float = 60.0
    translation_prior_weight: float = 160.0
    optimizer_maxiter: int = 60
    min_projected_points: int = 120
    accepted_delta_rotation_deg: float = 1.0
    accepted_delta_translation_m: float = 0.04
    min_objective_improvement: float = 0.5


@dataclass(frozen=True)
class NuScenesBenchmarkConfig:
    info_path: str
    data_root: str | None = None
    camera_names: tuple[str, ...] = ("CAM_FRONT",)
    sample_limit: int | None = None
    sample_tokens: tuple[str, ...] = ()
    methods: tuple[str, ...] = ("identity", "edge_refine", "oracle_gt")
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
    image_distance_transform: np.ndarray
    camera_matrix: np.ndarray
    lidar_points_xyz: np.ndarray
    lidar_points_intensity: np.ndarray


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
                metadata={
                    "sample_data_token": camera_payload.get("sample_data_token"),
                    "cam2ego": camera_payload.get("cam2ego"),
                    "lidar2ego": lidar_points.get("lidar2ego"),
                    "num_pts_feats": lidar_points.get("num_pts_feats"),
                    "metainfo": payload.get("metainfo"),
                },
            )
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
        "requested_sample_token_count": len(wanted_tokens),
        "loaded_sample_count": len(samples),
        "loaded_tokens_preview": seen_tokens[:20],
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
    distance_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    xyz, intensity = _load_lidar_points(sample.lidar_path)
    finite_mask = np.all(np.isfinite(xyz), axis=1) & np.isfinite(intensity)
    xy_norm = np.linalg.norm(xyz[:, :2], axis=1)
    finite_mask &= xy_norm <= float(config.max_point_xy_range_m)
    xyz = xyz[finite_mask]
    intensity = intensity[finite_mask]
    if xyz.size == 0:
        raise RuntimeError(f"All lidar points were filtered out for {sample.token}")
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
        image_distance_transform=distance_transform,
        camera_matrix=camera_matrix,
        lidar_points_xyz=xyz,
        lidar_points_intensity=intensity,
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


def _edge_alignment_cost(
    params: np.ndarray,
    *,
    context: EdgeAlignmentContext,
    initial_transform: np.ndarray,
    config: EdgeRefinementConfig,
) -> float:
    delta = _delta_transform(params[:3], params[3:])
    transform = np.asarray(initial_transform, dtype=float) @ delta
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
    xy = np.round(uv).astype(np.int32)
    distances = context.image_distance_transform[xy[:, 1], xy[:, 0]]
    normalized_intensity = intensity / max(float(np.max(intensity)), 1.0)
    weights = normalized_intensity * (1.0 / np.clip(depths, 1.0, 60.0))
    weighted_distance = float(np.average(distances, weights=weights))
    regularized = (
        weighted_distance
        + float(config.rotation_prior_weight) * float(np.linalg.norm(params[:3]) ** 2)
        + float(config.translation_prior_weight)
        * float(np.linalg.norm(params[3:]) ** 2)
    )
    return regularized


def run_edge_refinement(
    *,
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

    def objective(params: np.ndarray) -> float:
        return _edge_alignment_cost(
            params,
            context=context,
            initial_transform=initial_transform,
            config=config,
        )

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

    best_result = None
    best_cost = float("inf")
    for seed in seeds:
        result = minimize(
            objective,
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
    initial_cost = objective(np.zeros(6, dtype=float))
    applied_rotation_deg = float(np.degrees(np.linalg.norm(best_result.x[:3])))
    applied_translation_m = float(np.linalg.norm(best_result.x[3:]))
    accepted_update = (
        float(initial_cost - best_result.fun) >= float(config.min_objective_improvement)
        and applied_rotation_deg <= float(config.accepted_delta_rotation_deg)
        and applied_translation_m <= float(config.accepted_delta_translation_m)
    )
    if not accepted_update:
        final_transform = np.asarray(initial_transform, dtype=float)
    return {
        "method": "edge_refine",
        "calibrated_transform": final_transform,
        "objective_before": float(initial_cost),
        "objective_after": float(best_result.fun if accepted_update else initial_cost),
        "optimizer_success": bool(best_result.success),
        "optimizer_status": int(best_result.status),
        "optimizer_message": (
            str(best_result.message)
            if accepted_update
            else (
                "fallback_to_initial_guess_due_to_delta_guard:" f"{best_result.message}"
            )
        ),
        "applied_delta_rotvec_rad": [float(value) for value in best_result.x[:3]],
        "applied_delta_translation_m": [float(value) for value in best_result.x[3:]],
        "accepted_update": bool(accepted_update),
        "applied_delta_rotation_deg_norm": applied_rotation_deg,
        "applied_delta_translation_m_norm": applied_translation_m,
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


def _render_overlay(
    *,
    context: EdgeAlignmentContext,
    transform: np.ndarray,
    output_path: Path,
    title: str,
    min_depth_m: float,
) -> str | None:
    uv, depths, _ = _project_points(
        transform,
        context.lidar_points_xyz,
        context.camera_matrix,
        context.image_bgr.shape,
        min_depth_m=min_depth_m,
    )
    if uv.shape[0] == 0:
        return None
    overlay = context.image_bgr.copy()
    colors = cv2.applyColorMap(
        np.clip(
            (depths / max(float(np.percentile(depths, 95)), 1e-6)) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    for index, (u, v) in enumerate(np.round(uv).astype(np.int32)):
        cv2.circle(
            overlay,
            (int(u), int(v)),
            1,
            tuple(int(channel) for channel in colors[index, 0]),
            thickness=-1,
        )
    blended = cv2.addWeighted(context.image_bgr, 0.72, overlay, 0.28, 0.0)
    cv2.putText(
        blended,
        title,
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)
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
        yaml.safe_dump(metrics_output, file, sort_keys=False)
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
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
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
    edge_refine = method_rows_by_name.get("edge_refine") or {}
    edge_sample_rows = [
        row for row in per_sample_rows if row["method"] == "edge_refine"
    ]
    oracle_rotation = oracle.get("mean_final_rotation_error_deg")
    oracle_translation = oracle.get("mean_final_translation_error_m")
    edge_refine_rotation = edge_refine.get("mean_final_rotation_error_deg")
    edge_refine_translation = edge_refine.get("mean_final_translation_error_m")
    identity_rotation = identity.get("mean_final_rotation_error_deg")
    identity_translation = identity.get("mean_final_translation_error_m")
    strict_success_rate = edge_refine.get("strict_success_rate", 0.0)
    optimizer_success_ratio = (
        sum(1 for row in edge_sample_rows if row.get("optimizer_success"))
        / max(len(edge_sample_rows), 1)
        if edge_sample_rows
        else 0.0
    )
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
            "name": "edge_refine_available",
            "status": "pass" if bool(edge_refine) else "warning",
            "severity": "required",
            "evidence": f"edge_refine_rows={edge_refine.get('record_count', 0)}",
            "action": (
                "Enable the experimental edge_refine path to evaluate actual "
                "calibration recovery."
            ),
        },
        {
            "name": "edge_refine_vs_identity_rotation",
            "status": (
                "pass"
                if edge_refine
                and identity
                and float(edge_refine_rotation) < float(identity_rotation)
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                f"edge_refine_mean_rotation={edge_refine_rotation}, "
                f"identity_mean_rotation={identity_rotation}"
            ),
            "action": (
                "Require the experimental method to improve rotation "
                "recovery over the perturbed-input baseline."
            ),
        },
        {
            "name": "edge_refine_vs_identity_translation",
            "status": (
                "pass"
                if edge_refine
                and identity
                and float(edge_refine_translation) < float(identity_translation)
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                f"edge_refine_mean_translation={edge_refine_translation}, "
                f"identity_mean_translation={identity_translation}"
            ),
            "action": (
                "Require the experimental method to improve translation "
                "recovery over the perturbed-input baseline."
            ),
        },
        {
            "name": "edge_refine_strict_success",
            "status": (
                "pass"
                if edge_refine and float(strict_success_rate) >= 0.50
                else "warning"
            ),
            "severity": "advisory",
            "evidence": f"strict_success_rate={strict_success_rate}",
            "action": (
                "Grow the benchmark and improve the objective until strict "
                "recovery succeeds on at least half the cases."
            ),
        },
        {
            "name": "optimizer_success",
            "status": (
                "pass"
                if edge_sample_rows and optimizer_success_ratio >= 0.90
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
            config=config.edge_refinement,
        )
    raise ValueError(f"Unsupported benchmark method: {method_name}")


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
    rng = np.random.default_rng(int(config.random_seed))
    for context_index, context in enumerate(contexts):
        sample = context.sample
        gt_transform = np.asarray(sample.lidar_to_camera, dtype=float)
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
                    )
                    final_transform = np.asarray(
                        outcome["calibrated_transform"], dtype=float
                    )
                    final_delta = transform_delta_metrics(gt_transform, final_transform)
                    row = {
                        "sample_token": sample.token,
                        "sample_idx": sample.sample_idx,
                        "camera_name": sample.camera_name,
                        "timestamp": sample.timestamp,
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
                        "optimizer_success": bool(outcome.get("optimizer_success")),
                        "optimizer_status": outcome.get("optimizer_status"),
                        "optimizer_message": outcome.get("optimizer_message"),
                        "accepted_update": outcome.get("accepted_update"),
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
                        )
                        if overlay_file is not None:
                            overlays.append(
                                {
                                    "sample_token": sample.token,
                                    "camera_name": sample.camera_name,
                                    "method": method_name,
                                    "bucket": bucket,
                                    "path": overlay_file,
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
