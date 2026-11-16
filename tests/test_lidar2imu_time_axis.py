from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lidar2imu.adapters.base import AdapterConfig
from lidar2imu.adapters.core_record import _read_interchange
from lidar2imu.algorithms.gril.full_runner import (
    write_full_native_config,
    write_full_native_input,
)
from lidar2imu.dataset_io import load_dataset, write_dataset
from lidar2imu.evaluation.comparison import compare_datasets
from lidar2imu.models import CanonicalDataset, ImuBatch, LidarBatch
from lidar2imu.time_axis import replay_hard_offset_ns

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / ".agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml"
SECOND = 1_000_000_000
YEAR_2000 = 946684800 * SECOND
YEAR_2023 = 1672531200 * SECOND
HOST_2026 = 1767225600 * SECOND


def capture(reverse_epoch=False, imu_first=True):
    lidar_epoch, imu_epoch = (
        (YEAR_2023, YEAR_2000) if reverse_epoch else (YEAR_2000, YEAR_2023)
    )
    starts = np.arange(3, dtype=np.int64) * 100_000_000
    imu_time = np.arange(41, dtype=np.int64) * 10_000_000
    imu_time += -20_000_000 if imu_first else 20_000_000
    count = 160
    angle = np.linspace(0, 2 * np.pi, count)
    xyz = np.column_stack(
        (5 * np.cos(angle), 5 * np.sin(angle), np.full(count, -0.6))
    ).astype(np.float32)
    lidar_dispatch = HOST_2026 + starts + 120_000_000
    imu_jitter = np.resize([-20_000_000, 20_000_000, 0], len(imu_time))
    imu_jitter[-2:] = 0
    return CanonicalDataset(
        source_type="synthetic",
        source_files=("capture.record",),
        lidar_topic="/lidar",
        imu_topic="/imu",
        lidar=LidarBatch(
            frame_id="lidar",
            scan_timestamps_ns=lidar_epoch + starts,
            scan_offsets=np.arange(4, dtype=np.int64) * count,
            xyz=np.tile(xyz, (3, 1)),
            intensity=np.arange(3 * count, dtype=np.float32),
            ring=np.tile(np.arange(count, dtype=np.uint16) % 16, 3),
            point_time_s=np.tile(np.linspace(0, 0.1, count, dtype=np.float32), 3),
            measurement_timestamps_ns=lidar_epoch + starts + 100_000_000,
            dispatch_timestamps_ns=lidar_dispatch,
        ),
        imu=ImuBatch(
            frame_id="imu",
            timestamps_ns=imu_epoch + imu_time,
            angular_velocity=np.arange(len(imu_time) * 3).reshape(-1, 3) * 0.001,
            linear_acceleration=np.tile([0, 0, 9.81], (len(imu_time), 1)),
            dispatch_timestamps_ns=HOST_2026 + imu_time + 20_000_000 + imu_jitter,
        ),
    )


def overlapping_capture(offset_ns, imu_first=True):
    source = capture(imu_first=imu_first)
    starts = np.array([0, 5, 10], dtype=np.int64) * SECOND
    imu_time = np.arange(1025, dtype=np.int64) * 10_000_000
    imu_time += -20_000_000 if imu_first else 20_000_000
    return replace(
        source,
        lidar=replace(
            source.lidar,
            scan_timestamps_ns=YEAR_2000 + starts,
            measurement_timestamps_ns=YEAR_2000 + starts + 100_000_000,
            dispatch_timestamps_ns=HOST_2026 + starts + 120_000_000,
        ),
        imu=replace(
            source.imu,
            timestamps_ns=YEAR_2000 + offset_ns + imu_time,
            angular_velocity=np.zeros((len(imu_time), 3)),
            linear_acceleration=np.tile([0, 0, 9.81], (len(imu_time), 1)),
            dispatch_timestamps_ns=HOST_2026 + imu_time + 20_000_000,
        ),
    )


def read_native(path):
    events = []
    with path.open("rb") as stream:
        assert stream.readline() == b"GRIL_NATIVE_DATASET 2\n"
        hard_offset = struct.unpack("<q", stream.read(8))[0]
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            event_type = stream.read(1)[0]
            if event_type == 0:
                timestamp, *values = struct.unpack("<q6d", stream.read(56))
                events.append((0, timestamp, values))
            else:
                _, timestamp, points = struct.unpack("<IqQ", stream.read(20))
                events.append((1, timestamp, stream.read(points * 22)))
        assert not stream.read()
    return hard_offset, events


class TimeAxisTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".time-axis-", dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_epoch_sign_startup_order_roundtrip_and_signal_preservation(self):
        for reverse in (False, True):
            for imu_first in (False, True):
                with self.subTest(reverse=reverse, imu_first=imu_first):
                    source = capture(reverse, imu_first)
                    manifest = write_dataset(source, self.root / "dataset")
                    loaded = load_dataset(manifest)
                    for left, right in (
                        (source.lidar, loaded.lidar),
                        (source.imu, loaded.imu),
                    ):
                        np.testing.assert_array_equal(
                            left.dispatch_timestamps_ns, right.dispatch_timestamps_ns
                        )
                    np.testing.assert_array_equal(
                        source.lidar.measurement_timestamps_ns,
                        loaded.lidar.measurement_timestamps_ns,
                    )
                    expected = (YEAR_2000 - YEAR_2023) * (1 if reverse else -1)
                    self.assertEqual(replay_hard_offset_ns(loaded), expected)
                    path = write_full_native_input(loaded, self.root / "native.bin")
                    hard_offset, events = read_native(path)
                    self.assertEqual(hard_offset, expected)
                    self.assertEqual(events[0][0], 0 if imu_first else 1)
                    aligned = [
                        timestamp - hard_offset if kind == 0 else timestamp
                        for kind, timestamp, _ in events
                    ]
                    self.assertEqual(aligned, sorted(aligned))
                    imu_events = [event for event in events if event[0] == 0]
                    np.testing.assert_array_equal(
                        [event[1] for event in imu_events], source.imu.timestamps_ns
                    )
                    np.testing.assert_array_equal(
                        [event[2] for event in imu_events],
                        np.column_stack(
                            (
                                source.imu.angular_velocity,
                                source.imu.linear_acceleration,
                            )
                        ),
                    )
                    point_dtype = np.dtype([("values", "<f4", (5,)), ("ring", "<u2")])
                    for index, event in enumerate(e for e in events if e[0] == 1):
                        points = np.frombuffer(event[2], dtype=point_dtype)
                        begin, end = source.lidar.scan_offsets[index : index + 2]
                        np.testing.assert_array_equal(
                            points["values"][:, 4], source.lidar.point_time_s[begin:end]
                        )
                    self.assertTrue(np.all(np.diff(aligned) >= 0))

    def test_scan_end_not_scan_start_and_reversed_headers(self):
        source = capture()
        self.assertTrue(np.any(np.diff(source.imu.dispatch_timestamps_ns) < 0))
        expected = replay_hard_offset_ns(source)
        wrong_reference = replace(
            source,
            lidar=replace(
                source.lidar,
                measurement_timestamps_ns=source.lidar.scan_timestamps_ns,
            ),
        )
        self.assertEqual(replay_hard_offset_ns(wrong_reference), expected + 100_000_000)
        # A reordered/delayed first header does not re-anchor that IMU sample.
        headers = source.imu.dispatch_timestamps_ns.copy()
        headers[0] += SECOND
        delayed = replace(
            source, imu=replace(source.imu, dispatch_timestamps_ns=headers)
        )
        self.assertGreater(headers[0], source.lidar.dispatch_timestamps_ns[0])
        self.assertEqual(replay_hard_offset_ns(delayed), expected)
        left = write_full_native_input(source, self.root / "left.bin").read_bytes()
        right = write_full_native_input(delayed, self.root / "right.bin").read_bytes()
        self.assertEqual(left, right)

    def test_overlapping_ranges_use_evidence_and_preserve_small_offset_baseline(self):
        for offset in (-5 * SECOND, -SECOND, 0, SECOND, 5 * SECOND):
            with self.subTest(offset=offset):
                source = overlapping_capture(offset)
                self.assertLessEqual(
                    max(
                        source.lidar.scan_timestamps_ns[0], source.imu.timestamps_ns[0]
                    ),
                    min(
                        source.lidar.scan_timestamps_ns[-1],
                        source.imu.timestamps_ns[-1],
                    ),
                )
                expected = offset if abs(offset) > SECOND else 0
                self.assertEqual(replay_hard_offset_ns(source), expected)

    def test_comparison_includes_time_values_and_presence_without_retiming(self):
        source = overlapping_capture(5 * SECOND)
        changed = replace(
            source,
            imu=replace(
                source.imu,
                dispatch_timestamps_ns=source.imu.dispatch_timestamps_ns + 2 * SECOND,
            ),
        )
        self.assertEqual(replay_hard_offset_ns(source), 5 * SECOND)
        self.assertEqual(replay_hard_offset_ns(changed), 3 * SECOND)
        comparison = compare_datasets(source, changed)
        self.assertEqual(comparison["verdict"], "different")
        self.assertTrue(comparison["checks"]["scan_timestamps"])
        self.assertTrue(comparison["checks"]["imu_timestamps"])
        self.assertFalse(comparison["checks"]["imu_dispatch_timestamps_ns"])
        for batch, name in (
            ("lidar", "measurement_timestamps_ns"),
            ("lidar", "dispatch_timestamps_ns"),
            ("imu", "dispatch_timestamps_ns"),
        ):
            key = f"{batch}_{name}"
            missing = replace(
                source, **{batch: replace(getattr(source, batch), **{name: None})}
            )
            for left, right in ((source, missing), (missing, source)):
                comparison = compare_datasets(left, right)
                self.assertEqual(comparison["verdict"], "different")
                self.assertEqual(
                    comparison["time_provenance"][key]["status"], "presence_mismatch"
                )
            legacy = compare_datasets(missing, missing)
            self.assertEqual(legacy["verdict"], "equivalent")
            self.assertEqual(legacy["time_provenance"][key]["status"], "both_absent")
        self.assertEqual(compare_datasets(source, source)["verdict"], "equivalent")
        np.testing.assert_array_equal(
            source.imu.timestamps_ns, changed.imu.timestamps_ns
        )
        self.assertEqual(replay_hard_offset_ns(source), 5 * SECOND)
        self.assertEqual(replay_hard_offset_ns(changed), 3 * SECOND)

    def test_legacy_same_clock_and_missing_cross_epoch_evidence(self):
        source = capture()
        legacy = replace(
            source,
            lidar=replace(
                source.lidar,
                measurement_timestamps_ns=None,
                dispatch_timestamps_ns=None,
            ),
            imu=replace(source.imu, dispatch_timestamps_ns=None),
        )
        with self.assertRaisesRegex(ValueError, "Disjoint sensor clocks"):
            write_full_native_input(legacy, self.root / "bad.bin")
        for name in ("dispatch_timestamps_ns", "measurement_timestamps_ns"):
            values = getattr(source.lidar, name).copy()
            values[1] = 0
            missing = replace(source, lidar=replace(source.lidar, **{name: values}))
            with self.assertRaisesRegex(ValueError, "complete independent dispatch"):
                replay_hard_offset_ns(missing)
        shared = replace(
            legacy,
            imu=replace(
                legacy.imu,
                timestamps_ns=legacy.imu.timestamps_ns - YEAR_2023 + YEAR_2000,
            ),
        )
        loaded = load_dataset(write_dataset(shared, self.root / "legacy"))
        self.assertIsNone(loaded.lidar.dispatch_timestamps_ns)
        self.assertEqual(replay_hard_offset_ns(loaded), 0)
        write_full_native_input(loaded, self.root / "legacy.bin")

    @unittest.skipUnless(
        os.environ.get("GRIL_NATIVE_FULL_TEST_EXECUTABLE"), "native opt-in"
    )
    def test_native_replay_packages_both_epoch_signs_orders_and_configured_lag(self):
        executable = Path(os.environ["GRIL_NATIVE_FULL_TEST_EXECUTABLE"]).resolve()
        for offset in (
            YEAR_2023 - YEAR_2000,
            YEAR_2000 - YEAR_2023,
            5 * SECOND,
            -5 * SECOND,
        ):
            for imu_first in (False, True):
                for configured in (-0.005, 0.005):
                    with self.subTest(offset=offset, imu_first=imu_first, C=configured):
                        source = (
                            capture(offset < 0, imu_first)
                            if abs(offset) > 5 * SECOND
                            else overlapping_capture(offset, imu_first)
                        )
                        boundary = write_full_native_input(
                            source, self.root / "data.bin"
                        )
                        config = write_full_native_config(
                            PROFILE,
                            self.root / "full.conf",
                            configured_time_lag_s=configured,
                        )
                        trace = self.root / "trace.txt"
                        result = subprocess.run(
                            [
                                str(executable),
                                "--input",
                                str(boundary),
                                "--config",
                                str(config),
                                "--trace",
                                str(trace),
                                "--output",
                                str(self.root / "result.txt"),
                                "--batch-trace",
                                str(self.root / "batch.txt"),
                                "--batch-executable",
                                str(executable.with_name("gril_native_batch")),
                                "--batch-config",
                                str(self.root / "unused.conf"),
                                "--gap-policy",
                                "reset",
                                "--forward-gap-s",
                                "0.05",
                            ],
                            cwd=self.root,
                            capture_output=True,
                            text=True,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(
                            "exhausted input before data_sufficiency", result.stderr
                        )
                        text = trace.read_text()
                        self.assertIn("package ", text)
                        self.assertIn("insufficient_data ", text)
                        self.assertIn(f"time_axis fixed {offset // SECOND} ", text)
                        self.assertNotIn("forward_gap_reset ", text)
                        imu_rows = [
                            row.split()
                            for row in text.splitlines()
                            if row.startswith("imu_event ")
                        ]
                        raw = np.array([float(row[2]) for row in imu_rows])
                        compensated = np.array([float(row[3]) for row in imu_rows])
                        hard_offset = replay_hard_offset_ns(source) / SECOND
                        np.testing.assert_allclose(
                            compensated,
                            raw - hard_offset - configured,
                            atol=5e-7,
                            rtol=0,
                        )
                        np.testing.assert_allclose(
                            np.diff(compensated), 0.01, atol=5e-7, rtol=0
                        )

    def test_interchange_v1_v2_optional_time_arrays(self):
        config = AdapterConfig(
            inputs=(self.root,), lidar_topic="/lidar", imu_topic="/imu"
        )
        for version in (1, 2):
            path = self.root / "canonical.bin"
            with path.open("wb") as stream:
                stream.write(f"GRIL_CANONICAL_NATIVE {version}\n".encode())
                for frame in (b"lidar", b"imu"):
                    stream.write(struct.pack("<I", len(frame)) + frame)
                stream.write(struct.pack("<4Q", 1, 1, 1, 0))
                stream.write(struct.pack("<q2Q", YEAR_2000, 0, 1))
                stream.write(struct.pack("<4fHf", 1, 2, 3, 4, 0, 0))
                stream.write(struct.pack("<q6d", YEAR_2023, 0, 0, 0, 0, 0, 9.81))
                if version == 2:
                    stream.write(
                        struct.pack("<3q", YEAR_2000 + SECOND, HOST_2026, HOST_2026)
                    )
            loaded = _read_interchange(path, config, ("fixture",))
            self.assertEqual(loaded.lidar.scan_timestamps_ns[0], YEAR_2000)
            if version == 2:
                self.assertEqual(
                    loaded.lidar.measurement_timestamps_ns[0], YEAR_2000 + SECOND
                )
                np.testing.assert_array_equal(
                    loaded.imu.dispatch_timestamps_ns, [HOST_2026]
                )
            else:
                self.assertIsNone(loaded.imu.dispatch_timestamps_ns)

    @unittest.skipUnless(
        os.environ.get("GRIL_RECORD_INGRESS_TEST_EXECUTABLE"), "record opt-in"
    )
    def test_real_record_cpp_to_canonical_roundtrip(self):
        from cyber_record.record import Record

        from lidar2imu.adapters.core_record import CoreApolloRecordAdapter
        from lidar2lidar.apollo_record_messages import get_message_class

        executable = Path(os.environ["GRIL_RECORD_INGRESS_TEST_EXECUTABLE"]).resolve()
        for reverse in (False, True):
            source = capture(reverse)
            path = self.root / f"capture_{reverse}.record"
            with Record(str(path), mode="w") as record:
                for index in range(3):
                    cloud = get_message_class("apollo.drivers.PointCloud")()
                    cloud.header.frame_id = "lidar"
                    cloud.header.timestamp_sec = (
                        source.lidar.dispatch_timestamps_ns[index] / SECOND
                    )
                    cloud.measurement_time = (
                        source.lidar.measurement_timestamps_ns[index] / SECOND
                    )
                    begin, end = source.lidar.scan_offsets[index : index + 2]
                    for offset in range(begin, end):
                        point = cloud.point.add()
                        point.x, point.y, point.z = source.lidar.xyz[offset]
                        point.intensity = int(source.lidar.intensity[offset])
                        point.timestamp = int(
                            source.lidar.scan_timestamps_ns[index]
                            + round(float(source.lidar.point_time_s[offset]) * SECOND)
                        )
                    record.write("/lidar", cloud, HOST_2026 + index * 100_000_000)
                for index, timestamp in enumerate(source.imu.timestamps_ns):
                    imu = get_message_class("apollo.drivers.gnss.Imu")()
                    imu.header.frame_id = "imu"
                    imu.header.timestamp_sec = (
                        source.imu.dispatch_timestamps_ns[index] / SECOND
                    )
                    imu.measurement_time = timestamp / SECOND
                    for axis, name in enumerate(("x", "y", "z")):
                        setattr(
                            imu.angular_velocity,
                            name,
                            source.imu.angular_velocity[index, axis],
                        )
                        setattr(
                            imu.linear_acceleration,
                            name,
                            source.imu.linear_acceleration[index, axis],
                        )
                    record.write("/imu", imu, HOST_2026 + index * 10_000_000)
            loaded = CoreApolloRecordAdapter(
                AdapterConfig(
                    inputs=(path,),
                    lidar_topic="/lidar",
                    imu_topic="/imu",
                    core_record_ingress=executable,
                )
            ).read()
            loaded = load_dataset(write_dataset(loaded, self.root / "decoded"))
            np.testing.assert_array_equal(
                loaded.lidar.scan_timestamps_ns, source.lidar.scan_timestamps_ns
            )
            np.testing.assert_allclose(
                loaded.imu.timestamps_ns, source.imu.timestamps_ns, atol=256, rtol=0
            )
            np.testing.assert_array_equal(
                loaded.imu.angular_velocity, source.imu.angular_velocity
            )
            self.assertLessEqual(
                abs(replay_hard_offset_ns(loaded) - replay_hard_offset_ns(source)), 256
            )
            boundary = write_full_native_input(loaded, self.root / "decoded.bin")
            hard_offset, events = read_native(boundary)
            aligned = [
                timestamp - hard_offset if kind == 0 else timestamp
                for kind, timestamp, _ in events
            ]
            self.assertEqual(aligned, sorted(aligned))
