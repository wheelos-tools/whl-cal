from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from lidar2imu.adapters.base import AdapterConfig
from lidar2imu.adapters.rosbag1 import Rosbag1Adapter
from lidar2imu.dataset_io import load_dataset, write_dataset


class Rosbag1AdapterTest(unittest.TestCase):
    def test_streams_point_arrays_and_writes_canonical_dataset(self) -> None:
        typestore = get_typestore(Stores.ROS1_NOETIC)
        types = {
            name: typestore.types[msgtype]
            for name, msgtype in {
                "Header": "std_msgs/msg/Header",
                "Time": "builtin_interfaces/msg/Time",
                "PointField": "sensor_msgs/msg/PointField",
                "PointCloud2": "sensor_msgs/msg/PointCloud2",
                "Quaternion": "geometry_msgs/msg/Quaternion",
                "Vector3": "geometry_msgs/msg/Vector3",
                "Imu": "sensor_msgs/msg/Imu",
            }.items()
        }
        point_dtype = np.dtype(
            {
                "names": ["x", "y", "z", "intensity", "time", "ring"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
                "offsets": [0, 4, 8, 12, 16, 20],
                "itemsize": 24,
            }
        )

        def stamp(timestamp_ns: int):
            return types["Time"](
                timestamp_ns // 1_000_000_000,
                timestamp_ns % 1_000_000_000,
            )

        def header(timestamp_ns: int, frame_id: str):
            return types["Header"](0, stamp(timestamp_ns), frame_id)

        def cloud(timestamp_ns: int, values: np.ndarray):
            fields = [
                types["PointField"](name, offset, datatype, 1)
                for name, offset, datatype in (
                    ("x", 0, 7),
                    ("y", 4, 7),
                    ("z", 8, 7),
                    ("intensity", 12, 7),
                    ("time", 16, 7),
                    ("ring", 20, 4),
                )
            ]
            return types["PointCloud2"](
                header(timestamp_ns, "lidar"),
                1,
                len(values),
                fields,
                False,
                point_dtype.itemsize,
                point_dtype.itemsize * len(values),
                values.view(np.uint8),
                False,
            )

        def imu(timestamp_ns: int):
            covariance = np.zeros(9, dtype=np.float64)
            return types["Imu"](
                header(timestamp_ns, "imu"),
                types["Quaternion"](0.0, 0.0, 0.0, 1.0),
                covariance,
                types["Vector3"](0.1, 0.2, 0.3),
                covariance,
                types["Vector3"](0.0, 0.0, 9.81),
                covariance,
            )

        first = np.zeros(3, dtype=point_dtype)
        first["x"] = [1.0, 2.0, np.nan]
        first["y"] = [0.0, 1.0, 2.0]
        first["z"] = [0.0, 0.0, 0.0]
        first["intensity"] = [10.0, 20.0, 30.0]
        first["time"] = [0.0, 0.01, 0.02]
        first["ring"] = [0, 1, 2]
        second = np.zeros(2, dtype=point_dtype)
        second["x"] = [3.0, 4.0]
        second["y"] = [0.0, 1.0]
        second["intensity"] = [30.0, 40.0]
        second["time"] = [0.0, 0.01]
        second["ring"] = [2, 3]

        with tempfile.TemporaryDirectory() as directory:
            bag_path = Path(directory) / "input.bag"
            with Writer(bag_path) as writer:
                lidar_connection = writer.add_connection(
                    "/lidar", "sensor_msgs/msg/PointCloud2", typestore=typestore
                )
                imu_connection = writer.add_connection(
                    "/imu", "sensor_msgs/msg/Imu", typestore=typestore
                )
                for timestamp_ns, message, connection, msgtype in (
                    (
                        1_000_000_000,
                        cloud(1_000_000_000, first),
                        lidar_connection,
                        "sensor_msgs/msg/PointCloud2",
                    ),
                    (
                        1_000_000_000,
                        imu(1_000_000_000),
                        imu_connection,
                        "sensor_msgs/msg/Imu",
                    ),
                    (
                        1_100_000_000,
                        cloud(1_100_000_000, second),
                        lidar_connection,
                        "sensor_msgs/msg/PointCloud2",
                    ),
                    (
                        1_100_000_000,
                        imu(1_100_000_000),
                        imu_connection,
                        "sensor_msgs/msg/Imu",
                    ),
                ):
                    writer.write(
                        connection,
                        timestamp_ns,
                        typestore.serialize_ros1(message, msgtype),
                    )

            dataset = Rosbag1Adapter(
                AdapterConfig(
                    inputs=(bag_path,),
                    lidar_topic="/lidar",
                    imu_topic="/imu",
                    scan_lines=4,
                )
            ).read()
            np.testing.assert_array_equal(dataset.lidar.scan_offsets, [0, 2, 4])
            np.testing.assert_array_equal(
                dataset.lidar.scan_timestamps_ns, [1_000_000_000, 1_100_000_000]
            )
            np.testing.assert_array_equal(
                dataset.lidar.xyz,
                [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 0.0], [4.0, 1.0, 0.0]],
            )
            np.testing.assert_array_equal(dataset.lidar.intensity, [10, 20, 30, 40])
            np.testing.assert_array_equal(dataset.lidar.ring, [0, 1, 2, 3])
            np.testing.assert_allclose(dataset.lidar.point_time_s, [0, 0.01, 0, 0.01])

            manifest = write_dataset(dataset, Path(directory) / "canonical")
            restored = load_dataset(manifest)
            np.testing.assert_array_equal(restored.lidar.xyz, dataset.lidar.xyz)
            np.testing.assert_array_equal(restored.lidar.ring, dataset.lidar.ring)


if __name__ == "__main__":
    unittest.main()
