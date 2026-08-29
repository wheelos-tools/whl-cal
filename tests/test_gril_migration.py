from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from gril.abtest import build_abtest_report
from gril.cli import build_parser
from gril.comparison import compare_datasets, compare_results, parse_gril_result
from gril.config import compare_configs, config_digest, load_algorithm_config
from gril.dataset_io import load_dataset, write_dataset
from gril.evaluation import build_input_contract
from gril.frontend_event import write_frontend_event_input
from gril.frontend_runner import NativeFrontendRunConfig, build_native_frontend_commands
from gril.frontend_trace import compare_frontend_traces
from gril.full_runner import (
    NativeFullRunConfig,
    build_full_native_command,
    write_full_native_config,
    write_full_native_input,
)
from gril.models import CanonicalDataset, ImuBatch, LidarBatch
from gril.native_runner import (
    NativeBatchRunConfig,
    build_native_command,
    native_config_values,
    write_native_config,
)
from gril.preprocess_trace import write_preprocess_input
from gril.reference import REFERENCE
from gril.reference_runner import ReferenceRunConfig, build_reference_command
from gril.validation import build_full_ab_review, write_full_ab_review


def _dataset() -> CanonicalDataset:
    return CanonicalDataset(
        source_type="synthetic",
        source_files=("synthetic",),
        lidar_topic="/velodyne_points",
        imu_topic="/imu/data",
        lidar=LidarBatch(
            frame_id="lidar",
            scan_timestamps_ns=np.array([1_000_000_000], dtype=np.int64),
            scan_offsets=np.array([0, 2], dtype=np.int64),
            xyz=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            intensity=np.array([1.0, 2.0], dtype=np.float32),
            ring=np.array([0, 1], dtype=np.uint16),
            point_time_s=np.array([0.0, 0.1], dtype=np.float32),
        ),
        imu=ImuBatch(
            frame_id="imu",
            timestamps_ns=np.array([900_000_000, 1_100_000_000], dtype=np.int64),
            angular_velocity=np.zeros((2, 3), dtype=np.float64),
            linear_acceleration=np.array(
                [[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]], dtype=np.float64
            ),
        ),
    )


class GrilDatasetTest(unittest.TestCase):
    def test_round_trip_and_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(_dataset(), Path(directory))
            loaded = load_dataset(manifest)
        comparison = compare_datasets(_dataset(), loaded)
        self.assertEqual(comparison["verdict"], "equivalent")

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(_dataset(), root)
            with (root / "imu.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_dataset(root)

    def test_input_contract_reports_timing_and_ring_coverage(self) -> None:
        contract = build_input_contract(_dataset())
        self.assertEqual(contract["verdict"], "accepted")
        self.assertEqual(contract["point_contract"]["ring_count"], 2)
        self.assertEqual(contract["counts"]["lidar_scans"], 1)

    def test_result_comparison(self) -> None:
        content = """
Rotation LiDAR to IMU = 1 2 3
Translation LiDAR to IMU = 0.1 0.2 0.3
Time Lag IMU to LiDAR = 0.004
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            path.write_text(content)
            result = parse_gril_result(path)
        comparison = compare_results(result, result)
        self.assertEqual(comparison["verdict"], "equivalent")

    def test_abtest_requires_input_and_result_equivalence(self) -> None:
        content = """
Rotation LiDAR to IMU = 1 2 3
Translation LiDAR to IMU = 0.1 0.2 0.3
Time Lag IMU to LiDAR = 0.004
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_dataset = root / "reference_dataset"
            candidate_dataset = root / "candidate_dataset"
            write_dataset(_dataset(), reference_dataset)
            write_dataset(_dataset(), candidate_dataset)
            reference_result = root / "reference.txt"
            candidate_result = root / "candidate.txt"
            reference_result.write_text(content)
            candidate_result.write_text(content)
            reference_trace = root / "reference_trace.txt"
            candidate_trace = root / "candidate_trace.txt"
            reference_trace.write_text("GRIL_BATCH_TRACE 1\n")
            candidate_trace.write_text("GRIL_BATCH_TRACE 1\n")
            config = root / "gril.yaml"
            config.write_text("""
preprocess: {lidar_type: 2}
calibration: {cut_frame: true}
mapping: {filter_size_surf: 0.5}
patchworkpp: {sensor_height: 1.0}
""")
            report = build_abtest_report(
                reference_dataset,
                candidate_dataset,
                reference_result,
                candidate_result,
                config,
                config,
                reference_trace,
                candidate_trace,
            )
        self.assertEqual(report["verdict"], "equivalent")
        self.assertNotIn("lidar2imu", str(report["methods"]))

    def test_abtest_rejects_same_trace_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_dataset(_dataset(), dataset)
            result = root / "result.txt"
            result.write_text(
                "Rotation LiDAR to IMU = 0 0 0\n"
                "Translation LiDAR to IMU = 0 0 0\n"
                "Time Lag IMU to LiDAR = 0\n"
            )
            trace = root / "trace.txt"
            trace.write_text("GRIL_BATCH_TRACE 1\n")
            config = root / "gril.yaml"
            config.write_text("""
preprocess: {}
calibration: {}
mapping: {}
patchworkpp: {}
launch: {}
""")
            with self.assertRaisesRegex(ValueError, "distinct run artifacts"):
                build_abtest_report(
                    dataset,
                    dataset,
                    result,
                    result,
                    config,
                    config,
                    trace,
                    trace,
                )

    def test_runtime_config_does_not_change_algorithm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.yaml"
            right = root / "right.yaml"
            algorithm = """
preprocess: {lidar_type: 2}
calibration: {cut_frame: true}
mapping: {filter_size_surf: 0.5}
patchworkpp: {sensor_height: 1.0}
"""
            left.write_text(algorithm + "common: {lid_topic: /reference}\n")
            right.write_text(algorithm + "common: {lid_topic: /native}\n")
            comparison = compare_configs(left, right)
        self.assertEqual(comparison["verdict"], "equivalent")

    def test_launch_only_odometry_values_change_algorithm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.yaml"
            right = root / "right.yaml"
            algorithm = """
cube_side_length: 1000
preprocess: {lidar_type: 2}
calibration: {cut_frame: true}
mapping: {filter_size_surf: 0.5}
patchworkpp: {sensor_height: 1.0}
"""
            left.write_text("max_iteration: 5\n" + algorithm)
            right.write_text("max_iteration: 4\n" + algorithm)
            comparison = compare_configs(left, right)
        self.assertEqual(comparison["verdict"], "different")

    def test_reviewed_config_pins_velodyne_launch_odometry_values(self) -> None:
        config_path = (
            Path(__file__).parents[1]
            / ".agents"
            / "skills"
            / "gril-calib-validation"
            / "resources"
            / "vanjeelidar16.yaml"
        )
        config = load_algorithm_config(config_path)
        self.assertEqual(config["launch"]["max_iteration"], 5)
        self.assertEqual(config["launch"]["cube_side_length"], 1000)

    def test_reference_runner_delegates_to_frozen_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "capture.record.00000"
            record.touch()
            command = build_reference_command(
                ReferenceRunConfig(
                    record_files=(record,),
                    output_dir=Path(directory) / "output",
                    lidar_topic="/lidar",
                    imu_topic="/imu",
                    pose_topic="/pose",
                    tf_parent="imu",
                    tf_child="lidar",
                    scan_lines=16,
                )
            )
        self.assertIn("gril_pipeline.py", " ".join(command))
        self.assertIn("all", command)
        self.assertEqual(command.count("--record-file"), 1)

    def test_native_config_preserves_gril_calibration_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gril.yaml"
            source.write_text("""
preprocess: {lidar_type: 2}
calibration:
  cut_frame: true
  imu_sensor_height: 0.527
  trans_IL_x: -0.34
  gyro_factor: 13.0
mapping: {filter_size_surf: 0.5}
patchworkpp: {sensor_height: 1.117}
""")
            values = native_config_values(source)
            native = write_native_config(source, root / "native.conf")
            native_content = native.read_text()
        self.assertEqual(values["imu_sensor_height"], 0.527)
        self.assertEqual(values["trans_IL_x"], -0.34)
        self.assertEqual(values["gyro_factor"], 13.0)
        self.assertIn("svd_threshold 0.01", native_content)

    def test_native_command_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "gril_native_batch"
            trace = root / "trace.txt"
            config = root / "gril.yaml"
            for path in (executable, trace, config):
                path.touch()
            output = root / "output"
            native_config = root / "native.conf"
            command = build_native_command(
                NativeBatchRunConfig(executable, trace, config, output),
                native_config,
            )
        self.assertEqual(command.count("--trace"), 1)
        self.assertEqual(command.count("--config"), 1)
        self.assertEqual(command.count("--output"), 1)

    def test_preprocess_input_selects_algorithm_boundary_scans(self) -> None:
        dataset = _dataset()
        lidar = dataset.lidar
        repeated = CanonicalDataset(
            source_type=dataset.source_type,
            source_files=dataset.source_files,
            lidar_topic=dataset.lidar_topic,
            imu_topic=dataset.imu_topic,
            lidar=LidarBatch(
                frame_id=lidar.frame_id,
                scan_timestamps_ns=np.arange(21, dtype=np.int64) + 1_000_000_000,
                scan_offsets=np.arange(22, dtype=np.int64) * 2,
                xyz=np.tile(lidar.xyz, (21, 1)),
                intensity=np.tile(lidar.intensity, 21),
                ring=np.tile(lidar.ring, 21),
                point_time_s=np.tile(lidar.point_time_s, 21),
            ),
            imu=dataset.imu,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(repeated, root / "dataset")
            config = root / "gril.yaml"
            config.write_text("""
preprocess:
  blind: 0.5
  point_filter_num: 3
  scan_line: 16
calibration: {cut_frame_num: 3}
mapping: {}
patchworkpp: {}
""")
            output = write_preprocess_input(
                root / "dataset",
                config,
                root / "preprocess.txt",
            )
            text = output.read_text()
        self.assertIn("scans 3", text)
        self.assertIn("scan 1 ", text)
        self.assertIn("scan 20 ", text)
        self.assertIn("scan 21 ", text)

    def test_frontend_trace_comparison_uses_machine_precision_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.txt"
            candidate = root / "candidate.txt"
            reference.write_text("state 1.0 1e-9\nEND\n")
            candidate.write_text(
                f"state {np.nextafter(1.0, 2.0)} " f"{np.nextafter(1e-9, 2e-9)}\nEND\n"
            )
            report = compare_frontend_traces(reference, candidate)
            candidate.write_text("state 1.0 1.1e-9\nEND\n")
            rejected = compare_frontend_traces(reference, candidate)
        self.assertEqual(report["verdict"], "equivalent")
        self.assertEqual(report["inexact_values"], 2)
        self.assertEqual(rejected["verdict"], "different")

    def test_frontend_event_input_requires_explicit_algorithm_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "gril.yaml"
            config.write_text("""
preprocess: {blind: 0.5}
calibration: {cut_frame_num: 3}
mapping: {}
patchworkpp: {}
""")
            with self.assertRaisesRegex(ValueError, "point_filter_num"):
                write_frontend_event_input(
                    root / "missing-dataset",
                    config,
                    root / "events.txt",
                )

    def test_native_frontend_commands_keep_stages_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_dataset(_dataset(), dataset)
            gril_config = root / "gril.yaml"
            gril_config.write_text("""
preprocess: {blind: 0.5, point_filter_num: 3, scan_line: 16}
calibration: {cut_frame_num: 3}
mapping: {}
patchworkpp: {}
""")
            frontend = root / "gril_native_frontend_event_trace"
            ground = root / "gril_native_ground_trace"
            frontend.touch()
            ground.touch()
            config = NativeFrontendRunConfig(
                dataset=dataset,
                gril_config=gril_config,
                frontend_event_executable=frontend,
                ground_executable=ground,
                output_dir=root / "output",
                scan_count=1,
            )
            commands = build_native_frontend_commands(
                config,
                root / "events.txt",
                root / "ground.txt",
            )
        self.assertEqual(len(commands), 2)
        self.assertIn("sync_trace.txt", commands[0][-2])
        self.assertEqual(commands[0][-1], "--all")
        self.assertIn("ground_trace.txt", commands[1][-1])

    def test_full_native_input_is_versioned_and_event_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(_dataset(), root / "dataset")
            output = write_full_native_input(dataset, root / "input.bin")
            with output.open("rb") as stream:
                self.assertEqual(
                    stream.readline(),
                    b"GRIL_NATIVE_DATASET 1\n",
                )
                self.assertEqual(
                    int.from_bytes(stream.read(8), "little"),
                    3,
                )
                self.assertEqual(stream.read(1), b"\x00")
                self.assertEqual(
                    int.from_bytes(stream.read(8), "little", signed=True),
                    900_000_000,
                )

    def test_full_native_input_keeps_trailing_imu_for_hard_offset_sync(self) -> None:
        dataset = _dataset()
        extended = CanonicalDataset(
            source_type=dataset.source_type,
            source_files=dataset.source_files,
            lidar_topic=dataset.lidar_topic,
            imu_topic=dataset.imu_topic,
            lidar=dataset.lidar,
            imu=ImuBatch(
                frame_id=dataset.imu.frame_id,
                timestamps_ns=np.array(
                    [900_000_000, 1_100_000_000, 3_100_000_000],
                    dtype=np.int64,
                ),
                angular_velocity=np.zeros((3, 3), dtype=np.float64),
                linear_acceleration=np.tile(np.array([[0.0, 0.0, 9.81]]), (3, 1)),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(extended, root / "dataset")
            output = write_full_native_input(manifest, root / "input.bin")
            with output.open("rb") as stream:
                stream.readline()
                event_count = int.from_bytes(stream.read(8), "little")
        self.assertEqual(event_count, 4)

    def test_full_native_config_pins_complete_frontend(self) -> None:
        config_path = (
            Path(__file__).parents[1]
            / ".agents"
            / "skills"
            / "gril-calib-validation"
            / "resources"
            / "vanjeelidar16.yaml"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = write_full_native_config(
                config_path,
                Path(directory) / "full.conf",
            )
            text = output.read_text()
        self.assertIn("GRIL_NATIVE_FULL_CONFIG 1", text)
        self.assertIn("preprocess 2 16 0.5 3 0 1 3", text)
        self.assertIn("mapping 5 1000.0 0.5 0.5", text)
        self.assertIn("runtime 0.0", text)

    def test_full_native_config_rejects_non_czm_or_invalid_zone_count(self) -> None:
        config_path = (
            Path(__file__).parents[1]
            / ".agents"
            / "skills"
            / "gril-calib-validation"
            / "resources"
            / "vanjeelidar16.yaml"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load(config_path.read_text())
            config["patchworkpp"]["mode"] = "unsupported"
            invalid_mode = root / "invalid_mode.yaml"
            invalid_mode.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "mode=czm"):
                write_full_native_config(invalid_mode, root / "full.conf")

            config["patchworkpp"]["mode"] = "czm"
            config["patchworkpp"]["czm"]["num_zones"] = 3
            invalid_zones = root / "invalid_zones.yaml"
            invalid_zones.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "four Patchwork"):
                write_full_native_config(invalid_zones, root / "full.conf")

    def test_full_native_command_separates_gap_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "dataset.yaml",
                "gril.yaml",
                "executable",
                "gril_native_batch",
                "input.bin",
            ):
                (root / name).touch()
            golden = NativeFullRunConfig(
                dataset=root / "dataset.yaml",
                gril_config=root / "gril.yaml",
                executable=root / "executable",
                output_dir=root / "output",
            )
            command = build_full_native_command(
                golden,
                root / "input.bin",
                root / "gril.yaml",
            )
            self.assertIn("golden", command)
            self.assertNotIn("--forward-gap-s", command)

            reset = NativeFullRunConfig(
                dataset=root / "dataset.yaml",
                gril_config=root / "gril.yaml",
                executable=root / "executable",
                output_dir=root / "output",
                gap_policy="reset",
                forward_gap_s=1.5,
            )
            command = build_full_native_command(
                reset,
                root / "input.bin",
                root / "gril.yaml",
            )
            self.assertEqual(command[-2:], ["--forward-gap-s", "1.5"])

    def test_reset_gap_policy_requires_explicit_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "dataset.yaml",
                "gril.yaml",
                "executable",
                "gril_native_batch",
            ):
                (root / name).touch()
            config = NativeFullRunConfig(
                dataset=root / "dataset.yaml",
                gril_config=root / "gril.yaml",
                executable=root / "executable",
                output_dir=root / "output",
                gap_policy="reset",
            )
            with self.assertRaisesRegex(ValueError, "forward_gap_s"):
                config.validate()

    def test_full_native_cli_exposes_batch_handoff(self) -> None:
        args = build_parser().parse_args(
            [
                "run-native",
                "--input",
                "dataset",
                "--input-type",
                "canonical",
                "--config",
                "gril.yaml",
                "--executable",
                "gril_native_full_frontend",
                "--batch-executable",
                "gril_native_batch",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.batch_executable, Path("gril_native_batch"))

    def test_full_review_keeps_scheduler_limitation_outside_result_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(_dataset(), root / "dataset")
            config = root / "gril.yaml"
            config.write_text("""
preprocess: {lidar_type: 2}
calibration: {cut_frame: true}
mapping: {filter_size_surf: 0.5}
patchworkpp: {sensor_height: 1.0}
""")
            result_template = (
                "Rotation LiDAR to IMU = {rotation} 0 0\n"
                "Translation LiDAR to IMU = 0 0 0\n"
                "Time Lag IMU to LiDAR = 0\n"
            )
            reference_result = root / "reference.txt"
            candidate_result = root / "candidate.txt"
            repeat_result = root / "repeat.txt"
            reference_result.write_text(result_template.format(rotation=0))
            candidate_result.write_text(result_template.format(rotation=0.25))
            repeat_result.write_text(result_template.format(rotation=0.25))
            candidate_trace = root / "candidate_trace.txt"
            candidate_trace.write_text(
                "package 1\npropagated 1\nupdated 1\nmotion_start 1\n"
                "calibration_push 1\nend_package 1\nbatch_handoff 1\n"
                "batch_complete 1\nEND\n"
            )
            reference_trace = root / "reference_trace.txt"
            reference_trace.write_text(
                "package 1\npropagated 1\nupdated 1\nmotion_start 1\n"
                "calibration_push 1\n"
            )
            native_config = root / "native.conf"
            native_config.write_text("native\n")
            manifest = {
                "input": {
                    "dataset": str(dataset),
                    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                },
                "algorithm_config": {
                    "digest": config_digest(load_algorithm_config(config)),
                    "native_sha256": "native-config",
                },
                "runtime_semantics": {"quality_gating_applied": False},
                "artifacts": {
                    "result": {"path": str(candidate_result)},
                    "full_frontend_trace": {
                        "path": str(candidate_trace),
                        "sha256": hashlib.sha256(
                            candidate_trace.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
            candidate_manifest = root / "candidate_manifest.yaml"
            candidate_manifest.write_text(yaml.safe_dump(manifest))
            manifest["artifacts"]["result"]["path"] = str(repeat_result)
            repeat_manifest = root / "repeat_manifest.yaml"
            repeat_manifest.write_text(yaml.safe_dump(manifest))
            patches = [
                {
                    "path": str(Path(REFERENCE[path_key]).resolve()),
                    "sha256": REFERENCE[hash_key],
                }
                for path_key, hash_key in (
                    ("validation_patch", "validation_patch_sha256"),
                    ("batch_trace_patch", "batch_trace_patch_sha256"),
                    ("preprocess_trace_patch", "preprocess_trace_patch_sha256"),
                    ("frontend_cv_trace_patch", "frontend_cv_trace_patch_sha256"),
                    ("ground_trace_patch", "ground_trace_patch_sha256"),
                    (
                        "full_frontend_reference_trace_patch",
                        "full_frontend_reference_trace_patch_sha256",
                    ),
                )
            ]
            archive = root / "archive.yaml"
            archive.write_text(
                yaml.safe_dump(
                    {
                        "source_revision": REFERENCE["revision"],
                        "source_archive_sha256": "trace-archive",
                        "patches": patches,
                        "traces": [
                            {
                                "path": str(reference_trace),
                                "sha256": hashlib.sha256(
                                    reference_trace.read_bytes()
                                ).hexdigest(),
                                "event_counts": {
                                    "package": 1,
                                    "propagated": 1,
                                    "updated": 1,
                                    "motion_start": 1,
                                    "calibration_push": 1,
                                },
                            }
                        ],
                    }
                )
            )
            bag = root / "reference.bag"
            bag.write_bytes(b"frozen input")
            bag_hash = hashlib.sha256(bag.read_bytes()).hexdigest()
            evidence = root / "evidence.yaml"
            evidence.write_text(
                yaml.safe_dump(
                    {
                        "schema": "GRIL_FULL_REVIEW_EVIDENCE 1",
                        "reviewed_at": "2026-08-29T16:55:50+08:00",
                        "reference_distribution": {
                            "sha256": REFERENCE["source_archive_sha256"]
                        },
                        "inputs": {
                            "frozen_reference_bag": str(bag),
                            "frozen_reference_bag_sha256": bag_hash,
                            "canonical_manifest_sha256": hashlib.sha256(
                                dataset.read_bytes()
                            ).hexdigest(),
                            "canonical_counts": {"lidar_scans": 1},
                            "canonical_array_hashes_verified": True,
                            "canonical_input_contract": "accepted",
                            "record_to_bag_contract": "accepted",
                        },
                        "proven_components": {
                            name: {"verdict": "equivalent"}
                            for name in (
                                "preprocess",
                                "fifo_synchronization",
                                "constant_velocity_propagation",
                                "patchwork_ground",
                                "isolated_odometry_ekf",
                            )
                        },
                        "full_frontend_non_bitwise_limitation": {
                            "verdict": "known_upstream_scheduler_nonreproducibility",
                            "source_behavior": "pthread background rebuild",
                        },
                        "physical_validation_separate_from_migration": {
                            "native_physical_holdout": "not_run",
                        },
                    }
                )
            )
            report = build_full_ab_review(
                reference_trace_archive_path=archive,
                reference_result_path=reference_result,
                reference_config_path=config,
                candidate_manifest_path=candidate_manifest,
                repeat_manifest_path=repeat_manifest,
                evidence_path=evidence,
            )
            review_path, summary_path = write_full_ab_review(report, root / "review")
            self.assertTrue(review_path.is_file())
            self.assertTrue(summary_path.is_file())

        self.assertEqual(
            report["verdict"]["full_runtime_reproduction"], "not_review_ready"
        )
        self.assertTrue(report["gates"]["proven_component_equivalence"])
        self.assertFalse(report["gates"]["complete_result_comparison"])
        self.assertTrue(report["gates"]["native_repeatability"])
        self.assertFalse(report["gates"]["exact_full_state_equality_required"])
        self.assertEqual(report["verdict"]["calibration_physical_quality"], "not_run")


if __name__ == "__main__":
    unittest.main()
