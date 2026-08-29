"""Command-line entrypoint for GRIL migration and A/B evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gril.abtest import build_abtest_report, write_abtest_report
from gril.adapters import ApolloRecordAdapter, Rosbag1Adapter
from gril.adapters.base import AdapterConfig
from gril.comparison import compare_datasets, compare_results, parse_gril_result
from gril.dataset_io import load_dataset, write_dataset
from gril.evaluation import write_input_contract
from gril.frontend_runner import NativeFrontendRunConfig, run_native_frontend
from gril.full_runner import NativeFullRunConfig, run_full_native
from gril.native_runner import NativeBatchRunConfig, run_native_batch
from gril.reference import REFERENCE
from gril.reference_runner import ReferenceRunConfig, run_reference
from gril.validation import build_full_ab_review, write_full_ab_review


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
    )
    adapter = (
        ApolloRecordAdapter(config)
        if args.input_type == "record"
        else Rosbag1Adapter(config)
    )
    dataset = adapter.read()
    manifest = write_dataset(dataset, args.output_dir)
    write_input_contract(dataset, args.output_dir)
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
    )
    adapter = (
        ApolloRecordAdapter(adapter_config)
        if args.input_type == "record"
        else Rosbag1Adapter(adapter_config)
    )
    dataset = adapter.read()
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
        )
        adapter = (
            ApolloRecordAdapter(adapter_config)
            if args.input_type == "record"
            else Rosbag1Adapter(adapter_config)
        )
        dataset = adapter.read()
        dataset_manifest = write_dataset(dataset, dataset_dir)
        write_input_contract(dataset, dataset_dir)

    result = run_full_native(
        NativeFullRunConfig(
            dataset=dataset_manifest,
            gril_config=args.config,
            executable=args.executable,
            output_dir=args.output_dir,
            batch_executable=args.batch_executable,
            scan_count=args.scan_count,
            gap_policy=args.gap_policy,
            forward_gap_s=args.forward_gap_s,
            configured_time_lag_s=args.configured_time_lag_s,
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
    parser.add_argument("--input-type", choices=("record", "bag"), required=True)
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
    parser.add_argument("--input-type", choices=("record", "bag"), default="record")
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
        choices=("record", "bag", "canonical"),
        default="record",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--batch-executable",
        type=Path,
        help="Defaults to gril_native_batch beside --executable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and compare ROS-independent GRIL migration inputs"
    )
    subparsers = parser.add_subparsers(required=True)
    _add_prepare(subparsers)
    _add_dataset_comparison(subparsers)
    _add_result_comparison(subparsers)
    _add_abtest(subparsers)
    _add_reference_run(subparsers)
    _add_native_batch_run(subparsers)
    _add_native_frontend_trace_run(subparsers)
    _add_native_full_run(subparsers)
    _add_full_review(subparsers)
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
