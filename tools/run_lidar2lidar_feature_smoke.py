#!/usr/bin/env python3
"""Focused regression smoke for L2L math, provenance and review artifacts."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from lidar2lidar import semantic_registration
from lidar2lidar.feature_trace import CHANNELS
from lidar2lidar.feature_visualization import (
    export_feature_index,
    export_feature_window,
)
from lidar2lidar.lie import information_left_to_right, skew
from lidar2lidar.semantic_registration import (
    LidarFrameData,
    PrimitiveExtractionConfig,
    SemanticRegistrationConfig,
    _tangent_basis,
    register_semantic_l2l,
)


def _numerical_math_smoke() -> None:
    frame = _scene()
    frame.intensity = np.ones(len(frame.points))
    frame.intensity[::80] = 1000.0
    rng = np.random.default_rng(41)
    target = LidarFrameData(
        frame.points + rng.normal(0.0, 0.003, frame.points.shape),
        intensity=frame.intensity,
    )
    seed = np.eye(4)
    seed[:3, :3] = Rotation.from_rotvec([0.015, -0.01, 0.025]).as_matrix()
    seed[:3, 3] = [0.13, -0.09, 0.04]
    config = SemanticRegistrationConfig(
        correspondence_schedule_m=(1.2,),
        uncertainty_schedule=((8.0, 1.2),),
        enable_coarse_fallback=False,
    )
    original_solver = semantic_registration.least_squares
    derivative_errors = []

    def checked_solver(fun, x0, **kwargs):
        for x in (
            np.zeros(6),
            np.array([1e-8, -2e-8, 1e-8, 0.1, -0.2, 0.3]),
            np.array([0.24, -0.17, 0.12, 0.3, -0.2, 0.1]),
            np.array([2.4, -0.7, 0.4, 0.3, -0.2, 0.1]),
        ):
            eps = 1e-6
            numeric = np.column_stack(
                [
                    (fun(x + eps * axis) - fun(x - eps * axis)) / (2.0 * eps)
                    for axis in np.eye(6)
                ]
            )
            analytic = kwargs["jac"](x)
            np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-7)
            derivative_errors.append(float(np.max(np.abs(analytic - numeric))))
        return original_solver(fun, x0, **kwargs)

    trace = {}
    with patch.object(semantic_registration, "least_squares", checked_solver):
        result = register_semantic_l2l(frame, target, seed, reg_cfg=config, trace=trace)
    assert result["success"]
    assert derivative_errors
    stage = trace["stages"][0]
    assert stage["optimized"]
    rotation = stage["T_after"][:3, :3]
    residuals = []
    jacobians = []
    for channel in CHANNELS:
        data = stage["channels"][channel]
        assert len(data["source_indices"]) > 0, channel
        points = trace["source"]["features"][channel]["points"][data["source_indices"]]
        vectors = trace["target"]["features"][channel]["vectors"][
            data["target_indices"]
        ]
        weight = getattr(config, f"{channel}_weight")
        components = data[
            "solver_residual_components_m" if channel == "pole" else "residual_vector_m"
        ]
        residuals.append(weight * components.reshape(-1))
        for point, vector in zip(points, vectors):
            basis = (
                _tangent_basis(vector)
                if channel == "pole"
                else np.eye(3) if channel == "intensity" else vector[None, :]
            )
            point_jacobian = np.column_stack([-rotation @ skew(point), rotation])
            jacobians.append(weight * basis @ point_jacobian)
    residual = np.concatenate(residuals)
    jacobian = np.vstack(jacobians)
    sigma = max(config.noise_floor_m, 1.4826 * np.median(np.abs(residual)))
    huber = np.minimum(1.0, config.huber_scale_m / np.maximum(np.abs(residual), 1e-12))
    expected_info = (jacobian.T * huber) @ jacobian / sigma**2
    np.testing.assert_allclose(result["info_matrix_6d"], expected_info, atol=1e-7)
    np.testing.assert_allclose(stage["info_matrix_6d"], expected_info, atol=1e-7)
    np.testing.assert_allclose(
        result["channel_rmse_m"]["overall_weighted"],
        np.sqrt(np.mean(residual**2)),
        atol=1e-12,
    )
    for block, direction_key in (
        (slice(0, 3), "weakest_rotation_direction_target"),
        (slice(3, 6), "weakest_translation_direction_target"),
    ):
        _, eigenvectors = np.linalg.eigh(expected_info[block, block])
        direction = result["localizability"][direction_key]
        np.testing.assert_allclose(
            abs(direction @ (rotation @ eigenvectors[:, 0])), 1.0
        )
    print(
        "PASS: additive Jacobians and solved-pose information "
        f"({max(derivative_errors):.2e})"
    )


def _joint_math_smoke() -> None:
    import open3d as o3d

    from lidar2imu.algorithms import joint_dual_calib as joint

    for xi in (
        np.array([0.0, 0.0, 0.0, 1.0, -2.0, 0.5]),
        np.array([1e-9, -2e-9, 3e-9, 1.0, -2.0, 0.5]),
        np.array([0.25, -0.17, 0.6, 1.0, -2.0, 0.5]),
        np.array([3.0, 0.4, -0.3, 1.0, -2.0, 0.5]),
    ):
        algebra = np.zeros((4, 4))
        algebra[:3, :3] = skew(xi[:3])
        algebra[:3, 3] = xi[3:]
        np.testing.assert_allclose(joint._se3_log6(expm(algebra)), xi, atol=1e-10)

    direction_frame = Rotation.from_euler("xyz", [25, 10, 70], degrees=True).as_matrix()
    localizability = semantic_registration._analyze_localizability(
        np.diag(np.arange(1.0, 7.0)),
        SemanticRegistrationConfig(),
        direction_frame,
    )
    for field in (
        "weakest_rotation_direction_target",
        "weakest_translation_direction_target",
    ):
        np.testing.assert_allclose(
            abs(np.dot(localizability[field], direction_frame[:, 0])), 1.0
        )

    rng = np.random.default_rng(71)
    points = rng.normal(size=(60, 3))
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([0.25, -0.12, 0.35]).as_matrix()
    transform[:3, 3] = [2.0, -3.0, 1.0]
    target_points = points @ transform[:3, :3].T + transform[:3, 3]
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    raw_info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, 0.001, transform
    )
    expected_jac = np.vstack(
        [
            np.column_stack([-transform[:3, :3] @ skew(point), transform[:3, :3]])
            for point in points
        ]
    )
    right_info = information_left_to_right(raw_info, transform)
    np.testing.assert_allclose(right_info, expected_jac.T @ expected_jac, atol=1e-10)
    whitener = joint._safe_cholesky_whitener(right_info)
    np.testing.assert_allclose(whitener.T @ whitener, right_info, atol=1e-10)

    xi = np.array([0.08, -0.05, 0.12, 0.1, -0.2, 0.05])
    algebra = np.zeros((4, 4))
    algebra[:3, :3] = skew(xi[:3])
    algebra[:3, 3] = xi[3:]
    screened = {
        "T_I_A_init": np.eye(4),
        "T_I_B_init": transform @ expm(algebra),
        "gravity_windows": [],
        "tau_a_init": 0.0,
        "tau_b_init": 0.0,
    }
    original_solver = joint.least_squares
    expected_common_spectrum = []
    expected_differential_spectrum = []

    def checked_solver(fun, x0, **kwargs):
        np.testing.assert_allclose(fun(x0)[12:18], whitener @ xi, atol=1e-10)
        sol = original_solver(fun, x0, **kwargs)
        transforms = []
        for offset, key in ((0, "T_I_A_init"), (6, "T_I_B_init")):
            transform_at_solution = screened[key].copy()
            transform_at_solution[:3, :3] = (
                Rotation.from_rotvec(sol.x[offset : offset + 3]).as_matrix()
                @ screened[key][:3, :3]
            )
            transform_at_solution[:3, 3] += sol.x[offset + 3 : offset + 6]
            transforms.append(transform_at_solution)

        def local_residual(delta):
            x = sol.x.copy()
            for offset, key, current in (
                (0, "T_I_A_init", transforms[0]),
                (6, "T_I_B_init", transforms[1]),
            ):
                algebra = np.zeros((4, 4))
                algebra[:3, :3] = skew(delta[offset : offset + 3])
                algebra[:3, 3] = delta[offset + 3 : offset + 6]
                new = expm(algebra) @ current
                x[offset : offset + 3] = Rotation.from_matrix(
                    new[:3, :3] @ screened[key][:3, :3].T
                ).as_rotvec()
                x[offset + 3 : offset + 6] = new[:3, 3] - screened[key][:3, 3]
            return fun(x)

        eps = 1e-6
        local_jac = np.column_stack(
            [
                (local_residual(eps * axis) - local_residual(-eps * axis)) / (2.0 * eps)
                for axis in np.eye(12)
            ]
        )
        Q = np.block([[np.eye(6), np.eye(6)], [-np.eye(6), np.eye(6)]]) / np.sqrt(2.0)
        assert np.linalg.norm(local_jac[12:18] @ Q.T[:, :6]) < 1e-6
        H_cd = Q @ local_jac.T @ local_jac @ Q.T
        expected_common_spectrum.extend(np.linalg.eigvalsh(H_cd[:6, :6]))
        expected_differential_spectrum.extend(np.linalg.eigvalsh(H_cd[6:, 6:]))
        return sol

    with patch.object(joint, "least_squares", checked_solver):
        solution = joint._solve_anisotropic_factor_graph(
            screened=screened,
            l2l_edges=[{"transformation": transform, "info_matrix_6d": right_info}],
            H_rot_a=np.eye(3) * 1e-6,
            H_trans_a=np.eye(3) * 1e-6,
            H_rot_b=np.eye(3) * 1e-6,
            H_trans_b=np.eye(3) * 1e-6,
            cfg=joint.JointCalibrationConfig(),
        )
    error = joint._se3_log6(np.linalg.inv(transform) @ solution["T_A_B"])
    assert np.linalg.norm(error) < 1e-5
    np.testing.assert_allclose(
        solution["observability"]["common_mode_6d_eigenvalues"],
        expected_common_spectrum,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        solution["observability"]["differential_mode_6d_eigenvalues"],
        expected_differential_spectrum,
        rtol=1e-5,
        atol=1e-6,
    )
    print("PASS: SE(3) log, Open3D/joint tangents, whitening and common-mode gauge")
    from lidar2lidar.lidar2lidar import (
        calibrate_lidar_extrinsic,
        perform_icp_registration,
    )

    with TestCase().assertRaises(ValueError):
        calibrate_lidar_extrinsic(None, None, method=99)
    with TestCase().assertRaises(ValueError):
        perform_icp_registration(None, None, np.eye(4), 0.25, 5, method=4)


def _joint_fallback_smoke(root: Path) -> None:
    import yaml

    from lidar2imu.algorithms import joint_dual_calib as joint

    empty = LidarFrameData(np.empty((0, 3)))
    screened = {
        "T_I_A_init": np.eye(4),
        "T_I_B_init": np.eye(4),
        "T_AB_init": np.eye(4),
        "tau_a_init": 0.0,
        "tau_b_init": 0.0,
        "gravity_windows": [],
        "turn_windows": [],
        "static_screening_log": [],
        "turn_screening_log": [],
        "static_windows": [
            {
                "window_id": "empty",
                "rel_time_s": 0.0,
                "frame_a": empty,
                "frame_b": empty,
            }
        ],
    }
    edge = {
        "window_id": "empty",
        "method": "gicp_v1",
        "transformation": np.eye(4),
        "info_matrix_6d": np.eye(6),
    }
    with (
        patch.multiple(
            joint,
            FastCanonicalScanAccess=Mock(
                return_value=SimpleNamespace(imu_frame="imu", lidar_frame="lidar")
            ),
            parse_detailed_gril_result=Mock(return_value={}),
            parse_batch_trace_arrays=Mock(return_value={}),
            screen_joint_calibration_windows=Mock(return_value=screened),
            _compute_l2i_anisotropic_info=Mock(return_value=(np.eye(3), np.eye(3))),
            _evaluate_l2l_edges_gicp=Mock(return_value=[edge]),
            evaluate_common_mode_excitation_gate=Mock(
                return_value={"verdict": "needs_more_excitation"}
            ),
        ),
        TestCase().assertLogs(joint.__name__, level="WARNING"),
    ):
        report = joint.run_joint_dual_calibration(
            dataset_a=root / "a.yaml",
            dataset_b=root / "b.yaml",
            run_dir_a=root,
            run_dir_b=root,
            output_dir=root,
            mode="v2",
        )
    assert report["selected_stage"] == "v1"
    assert "v2" not in report["stages"]
    assert report["semantic_fallback_to_gicp"]
    edges = yaml.safe_load((root / "diagnostics/l2l_edges.yaml").read_text())
    assert edges["v2_semantic_edges"] == []
    assert len(edges["v2_semantic_failures"]) == 1
    assert (
        edges["v2_semantic_failures"][0]["reason"]
        == "insufficient_structural_primitives"
    )
    print("PASS: explicit semantic failures and correctly labelled GICP fallback")


def _semantic_edge_weighting_smoke() -> None:
    from lidar2imu.algorithms import joint_dual_calib as joint

    empty = LidarFrameData(np.empty((0, 3)))
    raw_info = np.diag(np.arange(1.0, 7.0) * 100.0)
    screened = {
        "T_I_A_init": np.eye(4),
        "T_AB_init": np.eye(4),
        "static_windows": [
            {
                "window_id": f"weight_{i}",
                "rel_time_s": float(i),
                "frame_a": empty,
                "frame_b": empty,
            }
            for i in range(3)
        ],
    }
    results = []
    for i in range(3):
        transform = np.eye(4)
        if i == 2:
            transform[0, 3] = 1.0
        results.append(
            {
                "success": True,
                "transformation": transform,
                "channel_matches": {
                    "ground": 100,
                    "facade": 10 if i == 1 else 40,
                    "pole": 10,
                    "intensity": 0,
                },
                "channel_unique_targets": {
                    "ground": 50,
                    "facade": 10 if i == 1 else 40,
                    "pole": 5,
                    "intensity": 0,
                },
                "info_matrix_6d": raw_info.copy(),
                "information_tangent": "right_source",
                "sigma_eff_m": 0.025,
                "localizability": {
                    "eigenvalues_6d": np.diag(raw_info).tolist(),
                    "condition_number_6d": 6.0,
                },
                "directed_overlap": {},
                "channel_rmse_m": {},
            }
        )
    with patch.object(joint, "register_semantic_l2l", side_effect=results):
        edges = joint._evaluate_l2l_edges_semantic(
            screened, np.eye(4), joint.JointCalibrationConfig()
        )
    assert edges[1]["downweighted_by_weak_support"]
    assert edges[2]["consensus_outlier"]
    for edge in edges:
        np.testing.assert_allclose(
            edge["eigenvalues_6d"], np.linalg.eigvalsh(edge["info_matrix_6d"])
        )
        np.testing.assert_allclose(
            edge["registration_eigenvalues_6d"], np.diag(raw_info)
        )
    print("PASS: effective information spectra after weak-support/consensus weighting")


def _window_influence_smoke() -> None:
    from lidar2imu.algorithms import joint_dual_calib as joint

    screened = {
        "T_I_A_init": np.eye(4),
        "T_I_B_init": np.eye(4),
        "tau_a_init": 0.0,
        "tau_b_init": 0.0,
        "gravity_windows": [],
    }
    edges = []
    for i, x in enumerate((0.0, 0.0, 0.3)):
        transform = np.eye(4)
        transform[0, 3] = x
        edges.append(
            {
                "window_id": f"stop_{i}",
                "transformation": transform,
                "info_matrix_6d": np.eye(6) * 100,
                "downweighted_by_weak_support": i != 0,
            }
        )
    kwargs = {
        "screened": screened,
        "H_rot_a": np.eye(3) * 1e-8,
        "H_trans_a": np.eye(3) * 1e-8,
        "H_rot_b": np.eye(3) * 1e-8,
        "H_trans_b": np.eye(3) * 1e-8,
        "cfg": joint.JointCalibrationConfig(),
        "T_I_A_init": np.eye(4),
        "T_I_B_init": np.eye(4),
    }
    nominal = joint._solve_anisotropic_factor_graph(l2l_edges=edges, **kwargs)
    frozen_nominal = nominal["T_A_B"].copy()
    report = joint._evaluate_l2l_window_influence(
        l2l_edges=edges, solution=nominal, **kwargs
    )
    assert report["status"] == "review_only"
    assert "fewer_than_two_undownweighted_windows" in report["reasons"]
    assert report["undownweighted_window_ids"] == ["stop_0"]
    np.testing.assert_allclose(
        report["max_change_from_nominal"]["T_A_B"]["translation_m"],
        0.1,
        atol=1e-5,
    )
    np.testing.assert_array_equal(nominal["T_A_B"], frozen_nominal)
    np.testing.assert_array_equal(edges[2]["info_matrix_6d"], np.eye(6) * 100)
    with patch.object(joint, "_solve_anisotropic_factor_graph") as solver:
        single = joint._evaluate_l2l_window_influence(
            l2l_edges=edges[:1], solution=nominal, **kwargs
        )
    solver.assert_not_called()
    assert single["status"] == "insufficient_windows"
    assert single["max_change_from_nominal"] is None
    failed = dict(nominal, solver={"success": False, "status": 0})
    with patch.object(joint, "_solve_anisotropic_factor_graph", return_value=failed):
        incomplete = joint._evaluate_l2l_window_influence(
            l2l_edges=edges, solution=nominal, **kwargs
        )
    assert incomplete["status"] == "solver_failed"
    assert incomplete["max_change_from_nominal"] is None
    assert all(r["change_from_nominal"] is None for r in incomplete["windows"])
    with patch.object(joint, "_solve_anisotropic_factor_graph") as solver:
        unconverged = joint._evaluate_l2l_window_influence(
            l2l_edges=edges, solution=failed, **kwargs
        )
    solver.assert_not_called()
    assert unconverged["status"] == "nominal_solver_failed"
    assert unconverged["max_change_from_nominal"] is None
    print("PASS: fixed-factor window influence, missing redundancy and solver failure")


def _equal(a, b) -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _equal(left, right)
    else:
        assert a == b, (a, b)


def _scene() -> LidarFrameData:
    rng = np.random.default_rng(19)
    ground = np.column_stack(
        [
            rng.uniform(3, 14, 2000),
            rng.uniform(-5, 5, 2000),
            rng.normal(-1.5, 0.01, 2000),
        ]
    )
    wall = np.column_stack(
        [
            rng.uniform(3, 14, 4000),
            rng.normal(4.5, 0.01, 4000),
            rng.uniform(-1, 3, 4000),
        ]
    )
    poles = np.column_stack(
        [
            rng.normal(8.1, 0.004, 1200),
            rng.normal(1.1, 0.004, 1200),
            rng.uniform(-1, 3, 1200),
        ]
    )
    points = np.vstack([[np.nan, 0, 0], ground, wall, poles, [0, 0, 0]])
    return LidarFrameData(
        points=points,
        intensity=rng.uniform(1, 100, len(points)),
        ring=np.arange(len(points)) % 32,
        scan_indices=np.arange(len(points)) // 1000,
        point_indices=np.arange(len(points)) % 1000,
    )


def _line_context_smoke() -> None:
    y, z = np.meshgrid(np.arange(-2.04, 2.05, 0.51), np.arange(0.6, 4.4, 0.025))
    wall = LidarFrameData(
        np.column_stack([np.full(y.size, 8.05), y.ravel(), z.ravel()])
    )
    trace = {}
    result = semantic_registration.extract_geometric_primitives(wall, trace=trace)
    context = trace["line_context"]
    assert result.stats["planar_line_rejected_voxels"] > 0
    assert (
        context["rejected_as_planar"].sum()
        == result.stats["planar_line_rejected_voxels"]
    )
    central = (np.abs(result.pole_points[:, 1]) < 1.0) & (
        (result.pole_points[:, 2] > 1.6) & (result.pole_points[:, 2] < 3.0)
    )
    assert not np.any(central), "wall sampling columns are not physical lines"
    for index in np.flatnonzero(context["rejected_as_planar"]):
        start, end = context["support_offsets"][index : index + 2]
        support = wall.points[context["support_indices"][start:end]]
        centered = support - support.mean(axis=0)
        values = np.linalg.eigvalsh(centered.T @ centered / len(support))
        np.testing.assert_allclose(
            context["planarity"][index], (values[1] - values[0]) / values[2]
        )
        distance = abs(
            (context["points"][index] - support.mean(axis=0))
            @ context["normals"][index]
        )
        np.testing.assert_allclose(context["plane_distance_m"][index], distance)
        assert distance <= PrimitiveExtractionConfig().ground_inlier_dist_m
    theta, heights = np.meshgrid(
        np.linspace(0, 2 * np.pi, 20, endpoint=False), np.arange(0.6, 4.4, 0.025)
    )
    pole = LidarFrameData(
        np.column_stack(
            [
                12.1 + 0.08 * np.cos(theta.ravel()),
                1.1 + 0.08 * np.sin(theta.ravel()),
                heights.ravel(),
            ]
        )
    )
    actual = semantic_registration.extract_geometric_primitives(pole)
    assert len(actual.pole_points) >= 4
    assert actual.stats["planar_line_rejected_voxels"] == 0
    rng = np.random.default_rng(51)
    wall_near_pole = np.column_stack(
        [
            np.full(20000, 12.49),
            rng.uniform(-0.4, 2.6, 20000),
            rng.uniform(0.6, 4.4, 20000),
        ]
    )
    separated_pole = pole.points + np.array([0.75, 0, 0])
    near_wall = LidarFrameData(np.vstack([wall_near_pole, separated_pole]))
    adjacent_trace = {}
    adjacent = semantic_registration.extract_geometric_primitives(
        near_wall, trace=adjacent_trace
    )
    real_pole = (adjacent.pole_points[:, 0] > 12.7) & (
        (adjacent.pole_points[:, 2] > 1.6) & (adjacent.pole_points[:, 2] < 3.0)
    )
    assert np.any(real_pole), "a separate pole near a wall is not a wall strip"
    checked = adjacent_trace["line_context"]
    off_plane = (checked["points"][:, 0] > 12.7) & (
        checked["plane_distance_m"] > PrimitiveExtractionConfig().ground_inlier_dist_m
    )
    assert np.any(off_plane)
    assert not np.any(checked["rejected_as_planar"][off_plane])
    print("PASS: wall strips rejected; isolated and adjacent-wall poles retained")


def run_smoke() -> None:
    _line_context_smoke()
    _numerical_math_smoke()
    _joint_math_smoke()
    _semantic_edge_weighting_smoke()
    _window_influence_smoke()
    frame = _scene()
    config = PrimitiveExtractionConfig(
        max_ground_points=150,
        max_facade_points=45,
        max_pole_points=12,
    )
    seed = np.eye(4)
    seed[:3, 3] = [0.10, -0.08, 0.03]
    baseline = register_semantic_l2l(frame, frame, seed, prim_cfg=config)
    trace = {}
    result = register_semantic_l2l(frame, frame, seed, prim_cfg=config, trace=trace)
    _equal(baseline, result)
    assert result["success"]
    for side in ("source", "target"):
        for channel in CHANNELS:
            data = trace[side]["features"][channel]
            offsets = data["support_offsets"]
            assert len(offsets) == len(data["points"]) + 1
            for i, point in enumerate(data["points"]):
                support = data["support_indices"][offsets[i] : offsets[i + 1]]
                assert len(support)
                assert np.isfinite(frame.points[support]).all()
                np.testing.assert_allclose(frame.points[support].mean(axis=0), point)
    optimized = [s for s in trace["stages"] if s["optimized"]]
    for channel in CHANNELS:
        assert result["channel_unique_targets"][channel] == len(
            np.unique(optimized[-1]["channels"][channel]["target_indices"])
        )
    for channel in CHANNELS:
        assert len(optimized[-1]["channels"][channel]["source_indices"]) == (
            result["channel_matches"][channel]
        )
    for stage in optimized:
        for channel, data in stage["channels"].items():
            if channel == "pole":
                np.testing.assert_allclose(
                    data["residual_norm_m"],
                    np.linalg.norm(data["solver_residual_components_m"], axis=1),
                    atol=1e-12,
                )
            assert np.all(
                (data["huber_component_weights"] > 0)
                & (data["huber_component_weights"] <= 1)
            )
    for channel, data in optimized[0]["channels"].items():
        assert ("shared_visibility" in data["gates"]) == (
            channel in {"ground", "facade"}
        )
    far_seed = np.eye(4)
    far_seed[0, 3] = 200.0
    fail_trace = {}
    failure = register_semantic_l2l(
        frame,
        frame,
        far_seed,
        trace=fail_trace,
        reg_cfg=SemanticRegistrationConfig(enable_coarse_fallback=False),
    )
    assert failure["reason"] == "insufficient_covisible_correspondences"
    assert not fail_trace["stages"][0]["optimized"]
    empty_trace = {}
    empty = LidarFrameData(np.empty((0, 3)))
    empty_result = register_semantic_l2l(empty, empty, np.eye(4), trace=empty_trace)
    assert empty_result["reason"] == "insufficient_structural_primitives"
    rng = np.random.default_rng(21)
    noisy = LidarFrameData(frame.points + rng.normal(0, 0.003, frame.points.shape))
    late_trace = {}
    late_cfg = SemanticRegistrationConfig(
        correspondence_schedule_m=(1.2, 1e-14),
        uncertainty_schedule=((8, 1.2), (1.2, 0.12)),
        enable_coarse_fallback=False,
    )
    late = register_semantic_l2l(
        frame,
        noisy,
        np.eye(4),
        trace=late_trace,
        reg_cfg=late_cfg,
        prim_cfg=PrimitiveExtractionConfig(max_pole_points=0),
    )
    assert late["success"]
    assert late_trace["stages"][0]["optimized"]
    assert not late_trace["stages"][-1]["optimized"]
    with tempfile.TemporaryDirectory(prefix="l2l-feature-smoke-") as temp:
        root = Path(temp)
        reports = []
        for name, current_trace, current_result in (
            ("good", trace, result),
            ("failed", fail_trace, failure),
            ("empty", empty_trace, empty_result),
            ("late", late_trace, late),
        ):
            report = export_feature_window(
                root / name,
                current_trace,
                current_result,
                {"window_id": name},
            )
            reports.append(report)
            for image in report["images"]:
                assert (root / name / image).read_bytes().startswith(b"\x89PNG")
            with np.load(root / name / "features.npz", allow_pickle=False) as values:
                assert "source_frame_points" in values
            with (root / name / "correspondences.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            for stage in current_trace["stages"]:
                for channel, data in stage["channels"].items():
                    selected = [
                        r
                        for r in rows
                        if r["stage"] == str(stage["index"])
                        and r["channel"] == channel
                        and r["participated_in_solve"] == "True"
                    ]
                    expected = len(data["source_indices"]) if stage["optimized"] else 0
                    assert len(selected) == expected
            page = (root / name / "index.html").read_text()
            payload = page.split('<script id="data" type="application/json">')[1]
            parsed = json.loads(payload.split("</script>")[0])
            assert parsed["metadata"]["window_id"] == name
        export_feature_index(
            root,
            reports,
            [
                {"accepted": True},
                {"accepted": False, "skip_reason": "sync_dt_exceeded"},
            ],
        )
        assert "sync_dt_exceeded" in (root / "index.html").read_text()
        _screening_rejection_smoke(root / "screening", frame)
        _joint_fallback_smoke(root / "fallback")
    print(
        "PASS: solver equivalence, support provenance, gates, residual weights, "
        "failed/empty/late-failure galleries and CSV participation"
    )


def _screening_rejection_smoke(root: Path, frame: LidarFrameData) -> None:
    from lidar2imu.algorithms.joint_dual_calib import (
        JointCalibrationConfig,
        _export_l2l_feature_debug,
        screen_joint_calibration_windows,
    )

    class Access:
        scan_t_s = np.array([0.0])
        imu_t_s = np.linspace(0, 0.1, 20)
        imu_w = np.zeros((20, 3))
        imu_a = np.zeros((20, 3))

        def read_scan_with_attributes(self, index):
            return (
                frame.points,
                np.zeros(len(frame.points)),
                frame.intensity,
                frame.ring,
            )

    access = Access()
    result = {
        "rotation_LI_matrix": np.eye(3),
        "translation_LI_m": np.zeros(3),
        "time_offset_s": 0.0,
        "acc_bias_m_s2": np.zeros(3),
        "gyro_bias_rad_s": np.zeros(3),
    }
    cfg = JointCalibrationConfig(min_static_shared_cells=1000000)
    batch = {"lid_t": np.empty(0)}
    baseline = screen_joint_calibration_windows(
        access,
        access,
        result,
        result,
        batch,
        batch,
        cfg,
        include_turn_windows=False,
    )
    rejected = []
    observed = screen_joint_calibration_windows(
        access,
        access,
        result,
        result,
        batch,
        batch,
        cfg,
        include_turn_windows=False,
        feature_rejections=rejected,
    )
    _equal(baseline, observed)
    assert len(rejected) == 1
    assert rejected[0]["frame_a"] is not None
    assert not observed["static_windows"]
    _export_l2l_feature_debug(root, [], rejected, [], observed, {})
    assert (root / "rejected_001/features.png").is_file()
    assert "solver_attempt_count: 0" in (root / "visualization_index.yaml").read_text()
    delayed = Access()
    delayed.scan_t_s = np.array([2.0])
    rejected_sync = []
    screened = screen_joint_calibration_windows(
        access,
        delayed,
        result,
        result,
        batch,
        batch,
        cfg,
        include_turn_windows=False,
        feature_rejections=rejected_sync,
    )
    assert rejected_sync[0]["frame_a"] is None
    _export_l2l_feature_debug(root / "sync", [], rejected_sync, [], screened, {})
    assert "input not read" in (root / "sync/index.html").read_text()
    assert not (root / "sync/rejected_001/index.html").exists()


if __name__ == "__main__":
    run_smoke()
