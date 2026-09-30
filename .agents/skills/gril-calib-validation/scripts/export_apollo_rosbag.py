#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import imu_payload, message_timestamp_ns

LIDAR_TOPIC = "/apollo/sensor/vanjeelidar/up/PointCloud2"
IMU_TOPIC = "/apollo/sensor/gnss/imu"
ROS_LIDAR_TOPIC = "/velodyne_points"
ROS_IMU_TOPIC = "/imu/data"
SCAN_LINES = 16

POINT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "time", "ring"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8, 12, 16, 20],
        "itemsize": 24,
    }
)


def _time_message(timestamp_ns: int, time_type):
    return time_type(
        sec=int(timestamp_ns // 1_000_000_000),
        nanosec=int(timestamp_ns % 1_000_000_000),
    )


def _pointcloud_message(msg, sequence: int, types):
    points = msg.point
    if not points:
        return None, None

    x = np.fromiter((point.x for point in points), np.float32, len(points))
    y = np.fromiter((point.y for point in points), np.float32, len(points))
    z = np.fromiter((point.z for point in points), np.float32, len(points))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    point_timestamps = np.fromiter(
        (point.timestamp for point in points), dtype=np.uint64, count=len(points)
    )
    timestamp_ns = int(point_timestamps.min())
    cloud = np.empty(int(valid.sum()), dtype=POINT_DTYPE)
    cloud["x"] = x[valid]
    cloud["y"] = y[valid]
    cloud["z"] = z[valid]
    intensity = np.fromiter(
        (point.intensity for point in points), np.float32, len(points)
    )
    cloud["intensity"] = intensity[valid]
    cloud["time"] = (point_timestamps[valid] - np.uint64(timestamp_ns)).astype(
        np.float64
    ) * 1e-9
    cloud["ring"] = (np.arange(len(points), dtype=np.uint16) % SCAN_LINES)[valid]

    fields = [
        types["PointField"]("x", 0, 7, 1),
        types["PointField"]("y", 4, 7, 1),
        types["PointField"]("z", 8, 7, 1),
        types["PointField"]("intensity", 12, 7, 1),
        types["PointField"]("time", 16, 7, 1),
        types["PointField"]("ring", 20, 4, 1),
    ]
    frame_id = str(
        getattr(msg, "frame_id", "")
        or getattr(getattr(msg, "header", None), "frame_id", "")
    ).strip()
    if not frame_id:
        raise ValueError("LiDAR message has no frame_id.")
    header = types["Header"](
        sequence, _time_message(timestamp_ns, types["Time"]), frame_id
    )
    output = types["PointCloud2"](
        header,
        1,
        len(cloud),
        fields,
        False,
        POINT_DTYPE.itemsize,
        POINT_DTYPE.itemsize * len(cloud),
        cloud.view(np.uint8).reshape(-1),
        False,
    )
    return timestamp_ns, output


def _imu_message(msg, timestamp_ns: int, sequence: int, types):
    imu = imu_payload(msg)
    header = types["Header"](
        sequence, _time_message(timestamp_ns, types["Time"]), "imu"
    )
    orientation_covariance = np.zeros(9, dtype=np.float64)
    orientation_covariance[0] = -1.0
    return types["Imu"](
        header,
        types["Quaternion"](0.0, 0.0, 0.0, 1.0),
        orientation_covariance,
        types["Vector3"](
            float(imu.angular_velocity.x),
            float(imu.angular_velocity.y),
            float(imu.angular_velocity.z),
        ),
        np.zeros(9, dtype=np.float64),
        types["Vector3"](
            float(imu.linear_acceleration.x),
            float(imu.linear_acceleration.y),
            float(imu.linear_acceleration.z),
        ),
        np.zeros(9, dtype=np.float64),
    )


def main() -> None:
    global SCAN_LINES
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-file", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lidar-topic", default=LIDAR_TOPIC)
    parser.add_argument("--imu-topic", default=IMU_TOPIC)
    parser.add_argument("--scan-lines", type=int, default=SCAN_LINES)
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    args = parser.parse_args()
    SCAN_LINES = args.scan_lines

    typestore = get_typestore(Stores.ROS1_NOETIC)
    type_names = {
        "Time": "builtin_interfaces/msg/Time",
        "Header": "std_msgs/msg/Header",
        "PointField": "sensor_msgs/msg/PointField",
        "PointCloud2": "sensor_msgs/msg/PointCloud2",
        "Quaternion": "geometry_msgs/msg/Quaternion",
        "Vector3": "geometry_msgs/msg/Vector3",
        "Imu": "sensor_msgs/msg/Imu",
    }
    types = {name: typestore.types[msgtype] for name, msgtype in type_names.items()}
    start_ns = None if args.start_sec is None else int(args.start_sec * 1e9)
    end_ns = None if args.end_sec is None else int(args.end_sec * 1e9)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {"lidar": 0, "imu": 0}
    with Writer(args.output) as writer:
        lidar_connection = writer.add_connection(
            ROS_LIDAR_TOPIC, type_names["PointCloud2"], typestore=typestore
        )
        imu_connection = writer.add_connection(
            ROS_IMU_TOPIC, type_names["Imu"], typestore=typestore
        )
        for record_file in args.record_file:
            with Record(record_file) as record:
                for (
                    topic,
                    payload,
                    type_name,
                    record_timestamp_ns,
                ) in record.read_raw_messages([args.lidar_topic, args.imu_topic]):
                    msg = record.decode_message(topic, payload, type_name)
                    if topic == args.lidar_topic:
                        timestamp_ns, output = _pointcloud_message(
                            msg, counts["lidar"], types
                        )
                        if output is None:
                            continue
                        connection = lidar_connection
                        msgtype = type_names["PointCloud2"]
                        counter = "lidar"
                    else:
                        timestamp_ns = message_timestamp_ns(
                            topic, msg, record_timestamp_ns
                        )
                        output = _imu_message(msg, timestamp_ns, counts["imu"], types)
                        connection = imu_connection
                        msgtype = type_names["Imu"]
                        counter = "imu"

                    if start_ns is not None and timestamp_ns < start_ns:
                        continue
                    if end_ns is not None and timestamp_ns >= end_ns:
                        continue
                    writer.write(
                        connection,
                        timestamp_ns,
                        typestore.serialize_ros1(output, msgtype),
                    )
                    counts[counter] += 1

    print(f"Wrote {args.output}: {counts['lidar']} LiDAR, {counts['imu']} IMU")


if __name__ == "__main__":
    main()
