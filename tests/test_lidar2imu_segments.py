from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lidar2imu.cli import build_parser
from lidar2imu.dataset_io import load_dataset, write_dataset
from lidar2imu.models import CanonicalDataset, ImuBatch, LidarBatch
from lidar2imu.segments import extract_segment


def dataset() -> CanonicalDataset:
    return CanonicalDataset(
        source_type="synthetic",
        source_files=("capture",),
        lidar_topic="/lidar",
        imu_topic="/imu",
        lidar=LidarBatch(
            frame_id="lidar",
            scan_timestamps_ns=np.array([100, 200, 300, 400]),
            scan_offsets=np.array([0, 2, 4, 6, 8]),
            xyz=np.arange(24, dtype=np.float32).reshape(8, 3),
            intensity=np.arange(8, dtype=np.float32),
            ring=np.arange(8, dtype=np.uint16),
            point_time_s=np.tile(np.array([0, 20e-9], dtype=np.float32), 4),
        ),
        imu=ImuBatch(
            frame_id="imu",
            timestamps_ns=np.arange(100, 451, 10),
            angular_velocity=np.arange(108, dtype=np.float64).reshape(-1, 3),
            linear_acceleration=np.zeros((36, 3)),
        ),
    )


class SegmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = dataset()
        self.manifest = write_dataset(self.source, self.root / "source")

    def extract(self, source: CanonicalDataset | None = None, **overrides):
        options = {
            "start_ns": 190,
            "end_ns": 321,
            "max_lidar_gap_ns": 110,
            "max_imu_gap_ns": 10,
            "parent_manifest": self.manifest,
        }
        options.update(overrides)
        return extract_segment(source or self.source, **options)

    def test_exact_slice_and_boundary_exclusion(self) -> None:
        segment = self.extract(end_ns=320)
        np.testing.assert_array_equal(segment.lidar.scan_timestamps_ns, [200])
        np.testing.assert_array_equal(segment.lidar.scan_offsets, [0, 2])
        np.testing.assert_array_equal(segment.lidar.xyz, self.source.lidar.xyz[2:4])
        np.testing.assert_array_equal(
            segment.imu.timestamps_ns, np.arange(190, 320, 10)
        )
        self.assertEqual(segment.metadata["segment"]["boundary_scans_excluded"], 1)
        self.assertEqual(segment.metadata["segment"]["source_scan_index_start"], 1)
        segment.validate()

    def test_rejects_gaps_and_missing_imu_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "LiDAR segment contains a gap"):
            self.extract(max_lidar_gap_ns=99)
        with self.assertRaisesRegex(ValueError, "IMU segment contains a gap"):
            self.extract(max_imu_gap_ns=9)
        imu = self.source.imu
        source = replace(
            self.source,
            imu=replace(
                imu,
                timestamps_ns=imu.timestamps_ns[11:],
                angular_velocity=imu.angular_velocity[11:],
                linear_acceleration=imu.linear_acceleration[11:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "IMU does not bracket"):
            self.extract(source)

    def test_rejects_duplicate_timestamps(self) -> None:
        times = self.source.imu.timestamps_ns.copy()
        times[11] = times[10]
        source = replace(self.source, imu=replace(self.source.imu, timestamps_ns=times))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.extract(source)

    def test_cli_writes_reusable_dataset_and_refuses_overwrite(self) -> None:
        output = self.root / "segment"
        args = build_parser().parse_args(
            [
                "prepare-segment",
                "--input",
                str(self.manifest),
                "--output-dir",
                str(output),
                "--start-ns",
                "190",
                "--end-ns",
                "321",
                "--max-lidar-gap-ns",
                "110",
                "--max-imu-gap-ns",
                "10",
            ]
        )
        args.handler(args)
        segment = load_dataset(output)
        np.testing.assert_array_equal(segment.lidar.scan_timestamps_ns, [200, 300])
        self.assertTrue((output / "input_contract.yaml").exists())
        with self.assertRaises(FileExistsError):
            args.handler(args)


if __name__ == "__main__":
    unittest.main()
