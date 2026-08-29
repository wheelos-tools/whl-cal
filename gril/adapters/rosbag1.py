"""ROS1 bag adapter implemented without a ROS runtime."""

from __future__ import annotations

from collections import Counter

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from gril.adapters.base import AdapterConfig, Scan, pack_imu, pack_lidar
from gril.models import CanonicalDataset, StaticTransform

_POINT_TYPES = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


def _stamp_ns(message, fallback_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _frame_id(message, fallback: str) -> str:
    header = getattr(message, "header", None)
    return str(getattr(header, "frame_id", "") or fallback)


def _point_dtype(message) -> np.dtype:
    endian = ">" if message.is_bigendian else "<"
    names = []
    formats = []
    offsets = []
    for field in message.fields:
        if field.datatype not in _POINT_TYPES:
            raise ValueError(
                f"Unsupported PointField datatype {field.datatype} for {field.name}"
            )
        names.append(field.name)
        scalar = np.dtype(endian + _POINT_TYPES[field.datatype])
        formats.append(scalar if field.count == 1 else (scalar, field.count))
        offsets.append(field.offset)
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": message.point_step,
        }
    )


def _point_array(message) -> np.ndarray:
    dtype = _point_dtype(message)
    raw = memoryview(message.data)
    if message.row_step == message.width * message.point_step:
        return np.frombuffer(raw, dtype=dtype, count=message.width * message.height)
    rows = []
    for row in range(message.height):
        start = row * message.row_step
        rows.append(np.frombuffer(raw[start:], dtype=dtype, count=message.width))
    return np.concatenate(rows)


def _transform(message) -> np.ndarray:
    translation = message.translation
    rotation = message.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    ).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


class Rosbag1Adapter:
    def __init__(self, config: AdapterConfig):
        self.config = config

    def read(self) -> CanonicalDataset:
        self.config.validate()
        typestore = get_typestore(Stores.ROS1_NOETIC)
        scans: list[Scan] = []
        imu_samples: list[tuple[int, np.ndarray, np.ndarray]] = []
        transforms: dict[tuple[str, str], StaticTransform] = {}
        lidar_frame = ""
        imu_frame = ""
        rejected: Counter[str] = Counter()
        topics = {
            self.config.lidar_topic,
            self.config.imu_topic,
            "/tf_static",
        }

        for bag_path in self.config.inputs:
            with Reader(bag_path) as reader:
                connections = [
                    connection
                    for connection in reader.connections
                    if connection.topic in topics
                ]
                for connection, bag_timestamp_ns, payload in reader.messages(
                    connections=connections
                ):
                    message = typestore.deserialize_ros1(payload, connection.msgtype)
                    if connection.topic == "/tf_static":
                        for item in message.transforms:
                            key = (item.header.frame_id, item.child_frame_id)
                            transforms[key] = StaticTransform(
                                parent_frame=key[0],
                                child_frame=key[1],
                                matrix=_transform(item.transform),
                            )
                        continue

                    timestamp_ns = _stamp_ns(message, int(bag_timestamp_ns))
                    if not self.config.includes(timestamp_ns):
                        continue
                    if connection.topic == self.config.imu_topic:
                        imu_frame = imu_frame or _frame_id(message, "imu")
                        imu_samples.append(
                            (
                                timestamp_ns,
                                np.array(
                                    [
                                        message.angular_velocity.x,
                                        message.angular_velocity.y,
                                        message.angular_velocity.z,
                                    ]
                                ),
                                np.array(
                                    [
                                        message.linear_acceleration.x,
                                        message.linear_acceleration.y,
                                        message.linear_acceleration.z,
                                    ]
                                ),
                            )
                        )
                        continue

                    points = _point_array(message)
                    required = {"x", "y", "z", "intensity", "time", "ring"}
                    missing = sorted(required - set(points.dtype.names or ()))
                    if missing:
                        raise ValueError(
                            "GRIL canonical PointCloud2 requires fields: "
                            + ", ".join(sorted(required))
                            + f"; missing: {', '.join(missing)}"
                        )
                    if len(points) == 0:
                        rejected["empty_lidar_scan"] += 1
                        continue
                    xyz = np.column_stack(
                        [points["x"], points["y"], points["z"]]
                    ).astype(np.float32)
                    intensity = np.asarray(points["intensity"], dtype=np.float32)
                    point_time_s = np.asarray(points["time"], dtype=np.float32)
                    finite = (
                        np.isfinite(xyz).all(axis=1)
                        & np.isfinite(intensity)
                        & np.isfinite(point_time_s)
                    )
                    if not finite.any():
                        rejected["no_finite_lidar_points"] += 1
                        continue
                    lidar_frame = lidar_frame or _frame_id(message, "lidar")
                    scans.append(
                        Scan(
                            timestamp_ns=timestamp_ns,
                            xyz=xyz[finite],
                            intensity=intensity[finite],
                            ring=np.asarray(points["ring"], dtype=np.uint16)[finite],
                            point_time_s=point_time_s[finite],
                        )
                    )

        dataset = CanonicalDataset(
            source_type="ros1_bag",
            source_files=tuple(str(path.resolve()) for path in self.config.inputs),
            lidar_topic=self.config.lidar_topic,
            imu_topic=self.config.imu_topic,
            lidar=pack_lidar(scans, lidar_frame or "lidar"),
            imu=pack_imu(imu_samples, imu_frame or "imu"),
            transforms=tuple(transforms.values()),
            metadata={
                "adapter": "Rosbag1Adapter",
                "reader": "rosbags",
                "ros_runtime_required": False,
                "rejected": dict(rejected),
                "point_time": {
                    "field": "time",
                    "unit": "seconds",
                    "reference": "PointCloud2 header stamp",
                },
            },
        )
        dataset.validate()
        return dataset
