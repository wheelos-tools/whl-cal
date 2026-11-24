"""Command-line entrypoint for GRIL migration and A/B evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lidar2imu.adapters import create_adapter, supported_input_types
from lidar2imu.adapters.base import AdapterConfig
from lidar2imu.algorithms.gril.frontend_runner import (
    NativeFrontendRunConfig,
    run_native_frontend,
)
from lidar2imu.algorithms.gril.full_runner import NativeFullRunConfig, run_full_native
from lidar2imu.algorithms.gril.native_runner import (
    NativeBatchRunConfig,
    run_native_batch,
)
from lidar2imu.algorithms.joint_dual_calib import (
    JointCalibrationConfig,
    run_joint_dual_calibration,
)
from lidar2imu.dataset_io import load_dataset, write_dataset
from lidar2imu.evaluation import write_input_contract
from lidar2imu.evaluation.abtest import build_abtest_report, write_abtest_report
from lidar2imu.evaluation.comparison import (
    compare_datasets,
    compare_results,
    parse_gril_result,
)
from lidar2imu.evaluation.dual_sensor_diagnostics import (
    evaluate_dual_lidar2imu_diagnostics,
)
from lidar2imu.evaluation.migration_review import (
    build_full_ab_review,
    write_full_ab_review,
)
from lidar2imu.evaluation.quality_gates import (
    evaluate_run_quality_gates,
    write_quality_gate_report,
)
from lidar2imu.reference.gril.identity import REFERENCE
from lidar2imu.reference.gril.runner import ReferenceRunConfig, run_reference
from lidar2imu.segments import extract_segment


def _write_yaml(value: dict, path: Path | None) -> None:
    text = yaml.safe_dump(value, sort_keys=False)
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(path)


def _prepare(args: argparse.Namespace) -> None:
    config = AdapterConfig(
        inputs=tuple(args.input),
        lidar_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        scan_lines=args.scan_lines,
        start_ns=None if args.start_sec is None else int(args.start_sec * 1e9),
        end_ns=None if args.end_sec is None else int(args.end_sec * 1e9),
        core_record_ingress=args.core_record_ingress,
    )
    dataset = create_adapter(args.input_type, config).read()
    manifest = write_dataset(dataset, args.output_dir)
    write_input_contract(dataset, args.output_dir)
    print(manifest)


def _prepare_segment(args: argparse.Namespace) -> None:
    manifest_path = args.input if args.input.is_file() else args.input / "dataset.yaml"
    if args.output_dir.exists():
        raise FileExistsError(f"Segment output already exists: {args.output_dir}")
    dataset = load_dataset(manifest_path)
    segment = extract_segment(
        dataset,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        max_lidar_gap_ns=args.max_lidar_gap_ns,
        max_imu_gap_ns=args.max_imu_gap_ns,
        parent_manifest=manifest_path,
    )
    manifest = write_dataset(segment, args.output_dir)
    write_input_contract(segment, args.output_dir)
    print(manifest)


def _compare_datasets(args: argparse.Namespace) -> None:
    result = compare_datasets(
        load_dataset(args.left),
        load_dataset(args.right),
        point_tolerance=args.point_tolerance,
        point_time_tolerance_s=args.point_time_tolerance_s,
        imu_tolerance=args.imu_tolerance,
    )
    _write_yaml(result, args.output)
    if result["verdict"] != "equivalent":
        raise SystemExit(2)


def _compare_results(args: argparse.Namespace) -> None:
    result = compare_results(
        parse_gril_result(args.reference),
        parse_gril_result(args.candidate),
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        translation_tolerance_m=args.translation_tolerance_m,
        time_tolerance_s=args.time_tolerance_s,
    )
    _write_yaml(result, args.output)
    if result["verdict"] != "equivalent":
        raise SystemExit(2)


def _abtest(args: argparse.Namespace) -> None:
    report = build_abtest_report(
        args.reference_dataset,
        args.candidate_dataset,
        args.reference_result,
        args.candidate_result,
        args.reference_config,
        args.candidate_config,
        args.reference_trace,
        args.candidate_trace,
        point_tolerance=args.point_tolerance,
        point_time_tolerance_s=args.point_time_tolerance_s,
        imu_tolerance=args.imu_tolerance,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
        translation_tolerance_m=args.translation_tolerance_m,
        time_tolerance_s=args.time_tolerance_s,
    )
    output = write_abtest_report(report, args.output_dir)
    print(output)
    if report["verdict"] != "equivalent":
        raise SystemExit(2)


def _run_reference(args: argparse.Namespace) -> None:
    run_reference(
        ReferenceRunConfig(
            record_files=tuple(args.record_file),
            output_dir=args.output_dir,
            lidar_topic=args.lidar_topic,
            imu_topic=args.imu_topic,
            pose_topic=args.pose_topic,
            tf_parent=args.tf_parent,
            tf_child=args.tf_child,
            scan_lines=args.scan_lines,
            runs=args.runs,
            workspace=args.workspace,
            image=args.image,
            skip_submap=args.skip_submap,
        )
    )


def _run_native_batch(args: argparse.Namespace) -> None:
    result = run_native_batch(
        NativeBatchRunConfig(
            executable=args.executable,
            trace=args.trace,
            gril_config=args.config,
            output_dir=args.output_dir,
        )
    )
    print(result)


def _run_native_frontend_trace(args: argparse.Namespace) -> None:
    dataset_dir = args.output_dir / "dataset"
    adapter_config = AdapterConfig(
        inputs=tuple(args.input),
        lidar_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        scan_lines=args.scan_lines,
        start_ns=None,
        end_ns=None,
        core_record_ingress=args.core_record_ingress,
    )
    dataset = create_adapter(args.input_type, adapter_config).read()
    manifest = write_dataset(dataset, dataset_dir)
    write_input_contract(dataset, dataset_dir)
    result = run_native_frontend(
        NativeFrontendRunConfig(
            dataset=manifest,
            gril_config=args.config,
            frontend_event_executable=args.frontend_event_executable,
            ground_executable=args.ground_executable,
            output_dir=args.output_dir / "frontend",
            scan_count=args.scan_count,
        )
    )
    print(result)


def _run_native_full(args: argparse.Namespace) -> None:
    if args.input_type == "canonical":
        if len(args.input) != 1:
            raise ValueError("canonical input accepts exactly one --input")
        dataset_manifest = args.input[0]
    else:
        dataset_dir = args.output_dir / "dataset"
        adapter_config = AdapterConfig(
            inputs=tuple(args.input),
            lidar_topic=args.lidar_topic,
            imu_topic=args.imu_topic,
            scan_lines=args.scan_lines,
            start_ns=None,
            end_ns=None,
            core_record_ingress=args.core_record_ingress,
        )
        dataset = create_adapter(args.input_type, adapter_config).read()
        dataset_manifest = write_dataset(dataset, dataset_dir)
        write_input_contract(dataset, dataset_dir)

    result = run_full_native(
        NativeFullRunConfig(
            dataset=dataset_manifest,
            gril_config=args.config,
            executable=args.executable,
            output_dir=args.output_dir,
            batch_executable=args.batch_executable,
            start_scan=args.start_scan,
            scan_count=args.scan_count,
            gap_policy=args.gap_policy,
            forward_gap_s=args.forward_gap_s,
            configured_time_lag_s=args.configured_time_lag_s,
            ikd_rebuild=args.ikd_rebuild,
            cut_mode=args.cut_mode,
            min_cut_points=args.min_cut_points,
            cut_frame_num=args.cut_frame_num,
            odom_ground_mode=args.odom_ground_mode,
            min_effective_features=args.min_effective_features,
            ahrs_imu_hz=args.ahrs_imu_hz,
            ahrs_dt_mode=args.ahrs_dt_mode,
            motion_start_min_speed_mps=args.motion_start_min_speed_mps,
            motion_start_min_omega_rad_s=args.motion_start_min_omega_rad_s,
            min_accum_duration_s=args.min_accum_duration_s,
            max_accum_duration_s=args.max_accum_duration_s,
            eof_policy=args.eof_policy,
        )
    )
    print(result)


def _review_full(args: argparse.Namespace) -> None:
    report = build_full_ab_review(
        reference_trace_archive_path=args.reference_trace_archive,
        reference_result_path=args.reference_result,
        reference_config_path=args.reference_config,
        candidate_manifest_path=args.candidate_manifest,
        repeat_manifest_path=args.repeat_manifest,
        evidence_path=args.evidence,
    )
    review, summary = write_full_ab_review(report, args.output_dir)
    print(review)
    print(summary)


def _add_prepare(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare", help="Convert record or ROS1 bag input to the canonical dataset"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--input-type", choices=supported_input_types(), required=True)
    parser.add_argument("--core-record-ingress", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/vanjeelidar/up/PointCloud2",
    )
    parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
    parser.add_argument("--scan-lines", type=int, default=16)
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    parser.set_defaults(handler=_prepare)


def _add_prepare_segment(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-segment", help="Extract a bounded canonical research segment"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--max-lidar-gap-ns", type=int, required=True)
    parser.add_argument("--max-imu-gap-ns", type=int, required=True)
    parser.set_defaults(handler=_prepare_segment)


def _add_dataset_comparison(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare-datasets", help="Compare two canonical adapter outputs"
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--point-tolerance", type=float, default=1e-6)
    parser.add_argument("--point-time-tolerance-s", type=float, default=1e-8)
    parser.add_argument("--imu-tolerance", type=float, default=1e-12)
    parser.set_defaults(handler=_compare_datasets)


def _add_result_comparison(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare-results", help="Compare frozen and ROS-free GRIL result files"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.2)
    parser.add_argument("--translation-tolerance-m", type=float, default=0.03)
    parser.add_argument("--time-tolerance-s", type=float, default=0.001)
    parser.set_defaults(handler=_compare_results)


def _add_abtest(subparsers) -> None:
    parser = subparsers.add_parser(
        "abtest", help="Build one frozen-reference versus native GRIL report"
    )
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-tolerance", type=float, default=1e-6)
    parser.add_argument("--point-time-tolerance-s", type=float, default=1e-8)
    parser.add_argument("--imu-tolerance", type=float, default=1e-12)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.2)
    parser.add_argument("--translation-tolerance-m", type=float, default=0.03)
    parser.add_argument("--time-tolerance-s", type=float, default=0.001)
    parser.set_defaults(handler=_abtest)


def _add_reference_run(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-reference",
        help="Run the frozen ROS-GRIL container reference",
    )
    parser.add_argument("--record-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/vanjeelidar/up/PointCloud2",
    )
    parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
    parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
    parser.add_argument("--tf-parent", default="imu")
    parser.add_argument("--tf-child", default="vanjeelidar_up")
    parser.add_argument("--scan-lines", type=int, default=16)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--workspace", type=Path, default=Path(".cache/gril-validation")
    )
    parser.add_argument("--image", default="gril-calib-validation:2026-08")
    parser.add_argument("--skip-submap", action="store_true")
    parser.set_defaults(handler=_run_reference)


def _add_native_batch_run(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-native-batch",
        help="Replay a reference batch trace through the ROS-free GRIL core",
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_run_native_batch)


def _add_native_frontend_trace_run(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-native-frontend-trace",
        help="Run only preprocessing/sync/Patchwork migration traces",
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--input-type", choices=supported_input_types(), default="record"
    )
    parser.add_argument("--core-record-ingress", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frontend-event-executable", type=Path, required=True)
    parser.add_argument("--ground-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-count", type=int, default=21)
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/vanjeelidar/up/PointCloud2",
    )
    parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
    parser.add_argument("--scan-lines", type=int, default=16)
    parser.set_defaults(handler=_run_native_frontend_trace)


def _add_native_full_run(subparsers) -> None:
    parser = subparsers.add_parser(
        "run-native-frontend",
        aliases=("run-native",),
        help="Run complete ROS-free GRIL from record, bag, or canonical input",
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--input-type",
        choices=(*supported_input_types(), "canonical"),
        default="record",
    )
    parser.add_argument("--core-record-ingress", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--batch-executable",
        type=Path,
        help="Defaults to gril_native_batch beside --executable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start-scan",
        type=int,
        default=0,
        help="0-based scan index to start window extraction; default 0",
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        help="Limit scans for deterministic debugging; default consumes all",
    )
    parser.add_argument(
        "--gap-policy",
        choices=("golden", "reset"),
        default="golden",
        help="golden preserves frozen semantics; reset is explicit production mode",
    )
    parser.add_argument(
        "--forward-gap-s",
        type=float,
        help="Required reset threshold when --gap-policy=reset",
    )
    parser.add_argument("--configured-time-lag-s", type=float, default=0.0)
    parser.add_argument(
        "--ikd-rebuild",
        choices=("async", "sync"),
        default="async",
    )
    parser.add_argument(
        "--cut-mode",
        choices=("equal_count", "equal_time"),
        default="equal_count",
    )
    parser.add_argument("--min-cut-points", type=int, default=0)
    parser.add_argument("--cut-frame-num", type=int)
    parser.add_argument(
        "--odom-ground-mode",
        choices=("golden", "disabled"),
        default="golden",
    )
    parser.add_argument("--min-effective-features", type=int, default=0)
    parser.add_argument("--ahrs-imu-hz", type=int, default=200)
    parser.add_argument(
        "--ahrs-dt-mode",
        choices=("package_reset", "continuous"),
        default="package_reset",
    )
    parser.add_argument("--motion-start-min-speed-mps", type=float, default=0.0)
    parser.add_argument("--motion-start-min-omega-rad-s", type=float, default=0.0)
    parser.add_argument("--min-accum-duration-s", type=float, default=0.0)
    parser.add_argument("--max-accum-duration-s", type=float, default=0.0)
    parser.add_argument(
        "--eof-policy",
        choices=("fail", "calibrate"),
        default="fail",
    )
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/vanjeelidar/up/PointCloud2",
    )
    parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
    parser.add_argument("--scan-lines", type=int, default=16)
    parser.set_defaults(handler=_run_native_full)


def _add_full_review(subparsers) -> None:
    parser = subparsers.add_parser(
        "review-full",
        help="Read completed GRIL artifacts and write a migration-only review",
    )
    parser.add_argument("--reference-trace-archive", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--repeat-manifest", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Read-only component and physical-validation evidence YAML",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=_review_full)


def _evaluate_quality_gates(args: argparse.Namespace) -> None:
    report = evaluate_run_quality_gates(
        args.run_dir,
        repeat_run_dirs=args.repeat_run_dir,
        seed_run_dirs=args.seed_run_dir,
        window_run_dirs=args.window_run_dir,
        holdout_batch_trace=args.holdout_batch_trace,
    )
    if args.output is not None:
        print(write_quality_gate_report(report, args.output))
    else:
        _write_yaml(report, None)


def _add_quality_gates(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate-quality-gates",
        help=(
            "Evaluate post-run data sufficiency, excitation, observability, "
            "repeatability, and holdout gates"
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeat-run-dir", type=Path, action="append")
    parser.add_argument("--seed-run-dir", type=Path, action="append")
    parser.add_argument("--window-run-dir", type=Path, action="append")
    parser.add_argument("--holdout-batch-trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=_evaluate_quality_gates)


def _evaluate_dual_diagnostics(args: argparse.Namespace) -> None:
    evaluate_dual_lidar2imu_diagnostics(
        dataset_a=args.dataset_a,
        run_dir_a=args.run_dir_a,
        dataset_b=args.dataset_b,
        run_dir_b=args.run_dir_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        static_time_s=args.static_time_s,
        turn_time_s=args.turn_time_s,
        extrinsic_a_yaml=args.extrinsic_a_yaml,
        extrinsic_b_yaml=args.extrinsic_b_yaml,
    )
    print(args.output_dir / "dual_diagnostics.yaml")


def _add_dual_diagnostics(subparsers) -> None:
    parser = subparsers.add_parser(
        "evaluate-dual-diagnostics",
        help=(
            "Run 5-pillar dual-LiDAR-to-IMU visual and quantitative diagnostics "
            "(colored PCDs, 2D slices, turn IMU submap, L2L loop, gravity, dynamics)"
        ),
    )
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-a", default="left_front")
    parser.add_argument("--label-b", default="right_back")
    parser.add_argument("--static-time-s", type=float)
    parser.add_argument("--turn-time-s", type=float)
    parser.add_argument("--extrinsic-a-yaml", type=Path)
    parser.add_argument("--extrinsic-b-yaml", type=Path)
    parser.set_defaults(handler=_evaluate_dual_diagnostics)


def _calibrate_dual_joint(args: argparse.Namespace) -> None:
    cfg = JointCalibrationConfig(
        z_cm_prior_m=args.z_cm_prior_m,
        max_static_windows=args.max_static_windows,
        max_turn_windows=args.max_turn_windows,
    )
    run_joint_dual_calibration(
        dataset_a=args.dataset_a,
        run_dir_a=args.run_dir_a,
        dataset_b=args.dataset_b,
        run_dir_b=args.run_dir_b,
        output_dir=args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        mode=args.mode,
        config=cfg,
        save_l2l_feature_debug=args.save_l2l_feature_debug,
    )
    print(args.output_dir / "joint_calibration_result.yaml")
    if args.save_l2l_feature_debug:
        print(args.output_dir / "diagnostics" / "l2l_features" / "index.html")


def _add_calibrate_dual_joint(subparsers) -> None:
    parser = subparsers.add_parser(
        "calibrate-dual-joint",
        help=(
            "Run staged joint 2x L2I + upgraded L2L spatiotemporal calibration "
            "(v1 anisotropic factor graph, v2 semantic/intensity 3D-FoV L2L, "
            "v3 turn-window IMU-preintegrated dual-LiDAR BA)"
        ),
    )
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--run-dir-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--run-dir-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-a", default="left_front")
    parser.add_argument("--label-b", default="right_back")
    parser.add_argument(
        "--mode",
        choices=("v1", "v2", "v3", "all"),
        default="v2",
    )
    parser.add_argument("--z-cm-prior-m", type=float)
    parser.add_argument("--max-static-windows", type=int, default=5)
    parser.add_argument("--max-turn-windows", type=int, default=4)
    parser.add_argument(
        "--save-l2l-feature-debug",
        action="store_true",
        help=(
            "Export detected primitives, actual stage correspondences and "
            "interactive/headless L2L galleries (requires v2/v3/all and matplotlib)"
        ),
    )
    parser.set_defaults(handler=_calibrate_dual_joint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and compare ROS-independent GRIL migration inputs"
    )
    subparsers = parser.add_subparsers(required=True)
    _add_prepare(subparsers)
    _add_prepare_segment(subparsers)
    _add_dataset_comparison(subparsers)
    _add_result_comparison(subparsers)
    _add_abtest(subparsers)
    _add_reference_run(subparsers)
    _add_native_batch_run(subparsers)
    _add_native_frontend_trace_run(subparsers)
    _add_native_full_run(subparsers)
    _add_full_review(subparsers)
    _add_quality_gates(subparsers)
    _add_dual_diagnostics(subparsers)
    _add_calibrate_dual_joint(subparsers)
    reference = subparsers.add_parser(
        "reference-info", help="Print the frozen ROS-GRIL reference identity"
    )
    reference.add_argument("--output", type=Path)
    reference.set_defaults(handler=lambda args: _write_yaml(REFERENCE, args.output))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
