#!/usr/bin/env python3
import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import (
    build_transform_graph,
    extract_tf_edges,
    imu_payload,
    lookup_transform,
    message_timestamp_ns,
)

parser = argparse.ArgumentParser()
parser.add_argument("--record-file", action="append", required=True)
parser.add_argument("--bag", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--lidar-topic", default="/apollo/sensor/vanjeelidar/up/PointCloud2"
)
parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
parser.add_argument("--tf-parent", default="imu")
parser.add_argument("--tf-child", default="vanjeelidar_up")
parser.add_argument("--scan-lines", type=int, default=16)
parser.add_argument("--sample-index", action="append", type=int)
args = parser.parse_args()
if args.scan_lines < 2:
    parser.error("--scan-lines must be at least 2")

RECORDS = args.record_file
BAG = args.bag
OUTPUT = args.output
LIDAR_TOPIC = args.lidar_topic
IMU_TOPIC = args.imu_topic


def percentile(values, quantiles=(0, 50, 95, 100)):
    return {
        f"p{quantile}": float(value)
        for quantile, value in zip(quantiles, np.percentile(values, quantiles))
    }


raw_clouds = []
raw_imus = []
sample_indices = set(args.sample_index or [0, 1, 137, 412, 686, 1030, 1372, 1373, 1374])
raw_cloud_samples = {}
raw_frame_ids = set()
for record_path in RECORDS:
    with Record(record_path) as record:
        for topic, message, record_timestamp_ns in record.read_messages(
            topics=[LIDAR_TOPIC, IMU_TOPIC]
        ):
            if topic == IMU_TOPIC:
                imu = imu_payload(message)
                raw_imus.append(
                    (
                        message_timestamp_ns(topic, message, int(record_timestamp_ns)),
                        np.array(
                            [
                                imu.angular_velocity.x,
                                imu.angular_velocity.y,
                                imu.angular_velocity.z,
                                imu.linear_acceleration.x,
                                imu.linear_acceleration.y,
                                imu.linear_acceleration.z,
                            ],
                            dtype=float,
                        ),
                    )
                )
                continue

            points = message.point
            raw_index = len(raw_clouds)
            raw_frame_ids.add(str(message.frame_id))
            raw_frame_ids.add(str(message.header.frame_id))
            frame_id = str(message.frame_id or message.header.frame_id).strip()
            if points:
                xyz = np.array(
                    [(point.x, point.y, point.z) for point in points], dtype=np.float32
                )
                timestamps = np.fromiter(
                    (point.timestamp for point in points),
                    dtype=np.uint64,
                    count=len(points),
                )
                intensity = np.fromiter(
                    (point.intensity for point in points),
                    dtype=np.float32,
                    count=len(points),
                )
                finite = np.isfinite(xyz).all(axis=1)
                minimum_timestamp_ns = int(timestamps.min())
                maximum_timestamp_ns = int(timestamps.max())
            else:
                xyz = np.empty((0, 3), dtype=np.float32)
                timestamps = np.empty(0, dtype=np.uint64)
                intensity = np.empty(0, dtype=np.float32)
                finite = np.empty(0, dtype=bool)
                minimum_timestamp_ns = None
                maximum_timestamp_ns = None
            raw_clouds.append(
                {
                    "raw_index": raw_index,
                    "record_path": record_path,
                    "record_timestamp_ns": int(record_timestamp_ns),
                    "frame_id": frame_id,
                    "header_timestamp_ns": int(
                        round(float(message.header.timestamp_sec) * 1e9)
                    ),
                    "point_count": int(len(points)),
                    "finite_count": int(finite.sum()),
                    "minimum_point_timestamp_ns": minimum_timestamp_ns,
                    "maximum_point_timestamp_ns": maximum_timestamp_ns,
                }
            )
            if raw_index in sample_indices and len(points):
                raw_cloud_samples[minimum_timestamp_ns] = {
                    "xyz": xyz[finite],
                    "intensity": intensity[finite],
                    "point_time_ns": timestamps[finite],
                }

typestore = get_typestore(Stores.ROS1_NOETIC)
converted_clouds = []
converted_imus = []
converted_cloud_samples = {}
ring_elevations = defaultdict(list)
with Reader(BAG) as reader:
    for connection, timestamp_ns, payload in reader.messages():
        message = typestore.deserialize_ros1(payload, connection.msgtype)
        if connection.topic == "/imu/data":
            converted_imus.append(
                (
                    int(timestamp_ns),
                    np.array(
                        [
                            message.angular_velocity.x,
                            message.angular_velocity.y,
                            message.angular_velocity.z,
                            message.linear_acceleration.x,
                            message.linear_acceleration.y,
                            message.linear_acceleration.z,
                        ],
                        dtype=float,
                    ),
                )
            )
            continue
        if connection.topic != "/velodyne_points":
            continue

        fields = {field.name: field.offset for field in message.fields}
        point_dtype = np.dtype(
            {
                "names": ["x", "y", "z", "intensity", "time", "ring"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
                "offsets": [
                    fields["x"],
                    fields["y"],
                    fields["z"],
                    fields["intensity"],
                    fields["time"],
                    fields["ring"],
                ],
                "itemsize": message.point_step,
            }
        )
        points = np.frombuffer(message.data, dtype=point_dtype)
        header_timestamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        time_values = points["time"].astype(float)
        time_differences = np.diff(time_values)
        converted_clouds.append(
            {
                "header_timestamp_ns": header_timestamp_ns,
                "frame_id": str(message.header.frame_id).strip(),
                "bag_timestamp_ns": int(timestamp_ns),
                "point_count": int(len(points)),
                "minimum_time_s": float(time_values.min()),
                "maximum_time_s": float(time_values.max()),
                "time_monotonic": bool(np.all(time_differences >= 0.0)),
                "finite": bool(
                    np.isfinite(
                        np.column_stack([points["x"], points["y"], points["z"]])
                    ).all()
                ),
                "first_time_s": float(time_values[0]),
                "last_time_s": float(time_values[-1]),
                "ring_min": int(points["ring"].min()),
                "ring_max": int(points["ring"].max()),
            }
        )
        if header_timestamp_ns in raw_cloud_samples:
            converted_cloud_samples[header_timestamp_ns] = {
                "xyz": np.column_stack([points["x"], points["y"], points["z"]]).astype(
                    np.float32
                ),
                "intensity": points["intensity"].astype(np.float32),
                "point_time_ns": np.rint(time_values * 1e9).astype(np.int64)
                + header_timestamp_ns,
            }
        if len(converted_clouds) <= 20:
            horizontal = np.hypot(points["x"], points["y"])
            elevation = np.degrees(np.arctan2(points["z"], horizontal))
            for ring in range(args.scan_lines):
                ring_elevations[ring].extend(
                    elevation[points["ring"] == ring][::100].tolist()
                )

raw_imu_timestamps = np.array([entry[0] for entry in raw_imus], dtype=np.int64)
converted_imu_timestamps = np.array(
    [entry[0] for entry in converted_imus], dtype=np.int64
)
raw_imu_values = np.array([entry[1] for entry in raw_imus])
converted_imu_values = np.array([entry[1] for entry in converted_imus])
raw_imu_by_timestamp = {int(timestamp_ns): values for timestamp_ns, values in raw_imus}
converted_imu_by_timestamp = {
    int(timestamp_ns): values for timestamp_ns, values in converted_imus
}
matched_imu_timestamps = sorted(
    set(raw_imu_by_timestamp) & set(converted_imu_by_timestamp)
)
missing_imu_timestamps = sorted(
    set(raw_imu_by_timestamp) - set(converted_imu_by_timestamp)
)
unexpected_imu_timestamps = sorted(
    set(converted_imu_by_timestamp) - set(raw_imu_by_timestamp)
)
matched_raw_imu_values = np.array(
    [raw_imu_by_timestamp[timestamp_ns] for timestamp_ns in matched_imu_timestamps]
)
matched_converted_imu_values = np.array(
    [
        converted_imu_by_timestamp[timestamp_ns]
        for timestamp_ns in matched_imu_timestamps
    ]
)

sample_checks = []
for timestamp_ns in sorted(set(raw_cloud_samples) & set(converted_cloud_samples)):
    raw = raw_cloud_samples[timestamp_ns]
    converted = converted_cloud_samples[timestamp_ns]
    sample_checks.append(
        {
            "timestamp_ns": int(timestamp_ns),
            "point_count_equal": bool(len(raw["xyz"]) == len(converted["xyz"])),
            "xyz_max_absolute_error": float(
                np.max(np.abs(raw["xyz"] - converted["xyz"]))
            ),
            "intensity_max_absolute_error": float(
                np.max(np.abs(raw["intensity"] - converted["intensity"]))
            ),
            "point_timestamp_max_absolute_error_ns": int(
                np.max(
                    np.abs(
                        raw["point_time_ns"].astype(np.int64)
                        - converted["point_time_ns"].astype(np.int64)
                    )
                )
            ),
        }
    )

raw_nonempty = [
    entry for entry in raw_clouds if entry["minimum_point_timestamp_ns"] is not None
]
raw_exportable = [entry for entry in raw_nonempty if entry["finite_count"] > 0]
converted_timestamp_set = {entry["header_timestamp_ns"] for entry in converted_clouds}
raw_frame_by_timestamp = {
    entry["minimum_point_timestamp_ns"]: entry["frame_id"] for entry in raw_exportable
}
frame_id_mismatches = [
    entry
    for entry in converted_clouds
    if raw_frame_by_timestamp.get(entry["header_timestamp_ns"]) != entry["frame_id"]
]
missing_exportable = [
    entry
    for entry in raw_exportable
    if entry["minimum_point_timestamp_ns"] not in converted_timestamp_set
]
first_converted_lidar_timestamp_ns = min(converted_timestamp_set)
last_converted_lidar_timestamp_ns = max(converted_timestamp_set)
missing_lidar_before_range = [
    entry
    for entry in missing_exportable
    if entry["minimum_point_timestamp_ns"] < first_converted_lidar_timestamp_ns
]
missing_lidar_after_range = [
    entry
    for entry in missing_exportable
    if entry["minimum_point_timestamp_ns"] > last_converted_lidar_timestamp_ns
]
missing_lidar_inside_range = [
    entry
    for entry in missing_exportable
    if first_converted_lidar_timestamp_ns
    <= entry["minimum_point_timestamp_ns"]
    <= last_converted_lidar_timestamp_ns
]
empty_frames = [entry for entry in raw_clouds if entry["point_count"] == 0]
all_invalid_frames = [
    entry
    for entry in raw_clouds
    if entry["point_count"] > 0 and entry["finite_count"] == 0
]

tf_edges = extract_tf_edges(RECORDS)
static_edges = [edge for edge in tf_edges if edge.is_static]
static_graph = build_transform_graph(static_edges)
sensor_transform = lookup_transform(static_graph, args.tf_child, args.tf_parent)
if sensor_transform is None:
    raise RuntimeError(
        "No static TF path from sensor frame "
        f"{args.tf_child!r} to IMU frame {args.tf_parent!r}"
    )

predecessors = {args.tf_child: None}
frames_to_visit = deque([args.tf_child])
while frames_to_visit and args.tf_parent not in predecessors:
    current_frame = frames_to_visit.popleft()
    for next_frame in sorted(static_graph.get(current_frame, {})):
        if next_frame not in predecessors:
            predecessors[next_frame] = current_frame
            frames_to_visit.append(next_frame)
frame_path = []
current_frame = args.tf_parent
while current_frame is not None:
    frame_path.append(current_frame)
    current_frame = predecessors.get(current_frame)
frame_path.reverse()
if not frame_path or frame_path[0] != args.tf_child:
    raise RuntimeError(
        "Could not reconstruct static TF path from "
        f"{args.tf_child!r} to {args.tf_parent!r}"
    )

scan_durations_ms = np.array(
    [entry["maximum_time_s"] * 1000.0 for entry in converted_clouds]
)
first_times = np.array([entry["first_time_s"] for entry in converted_clouds])
last_times = np.array([entry["last_time_s"] for entry in converted_clouds])
imu_initial = converted_imu_values[:100]

report = {
    "verdict": "accepted",
    "records": RECORDS,
    "bag": str(BAG),
    "message_counts": {
        "apollo_lidar": len(raw_clouds),
        "apollo_imu": len(raw_imus),
        "ros_lidar": len(converted_clouds),
        "ros_imu": len(converted_imus),
        "apollo_empty_lidar": len(empty_frames),
        "apollo_all_invalid_lidar": len(all_invalid_frames),
        "exportable_lidar_missing_from_ros": len(missing_exportable),
        "apollo_imu_missing_from_ros": len(missing_imu_timestamps),
        "ros_imu_missing_from_apollo": len(unexpected_imu_timestamps),
    },
    "dropped_lidar_frames": {
        "empty": empty_frames,
        "all_invalid": all_invalid_frames,
        "unexpected_missing_exportable": missing_exportable,
        "missing_before_ros_time_range_count": len(missing_lidar_before_range),
        "missing_inside_ros_time_range_count": len(missing_lidar_inside_range),
        "missing_after_ros_time_range_count": len(missing_lidar_after_range),
    },
    "lidar_frame_ids_seen": sorted(raw_frame_ids),
    "point_contract": {
        "ring_source": "point_index_mod_scan_lines",
        "all_converted_points_finite": all(
            entry["finite"] for entry in converted_clouds
        ),
        "all_converted_frames_time_monotonic": all(
            entry["time_monotonic"] for entry in converted_clouds
        ),
        "scan_duration_ms": percentile(scan_durations_ms),
        "first_point_time_zero_frame_count": int(np.sum(first_times == 0.0)),
        "last_point_is_max_time_frame_count": int(
            np.sum(last_times == scan_durations_ms / 1000.0)
        ),
        "all_frame_ids_match_apollo_source": not frame_id_mismatches,
        "ring_range": [
            min(entry["ring_min"] for entry in converted_clouds),
            max(entry["ring_max"] for entry in converted_clouds),
        ],
        "sampled_exact_conversion_checks": sample_checks,
    },
    "ring_elevation_deg": {
        int(ring): {
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
        }
        for ring, values in sorted(ring_elevations.items())
        if values
    },
    "imu_contract": {
        "timestamp_sequence_exact": bool(
            np.array_equal(raw_imu_timestamps, converted_imu_timestamps)
        ),
        "matched_timestamp_count": len(matched_imu_timestamps),
        "missing_timestamps_ns": missing_imu_timestamps,
        "unexpected_timestamps_ns": unexpected_imu_timestamps,
        "missing_before_first_ros_imu": int(
            sum(
                timestamp_ns < converted_imu_timestamps[0]
                for timestamp_ns in missing_imu_timestamps
            )
        ),
        "missing_after_last_ros_imu": int(
            sum(
                timestamp_ns > converted_imu_timestamps[-1]
                for timestamp_ns in missing_imu_timestamps
            )
        ),
        "missing_inside_ros_time_range": int(
            sum(
                converted_imu_timestamps[0]
                <= timestamp_ns
                <= converted_imu_timestamps[-1]
                for timestamp_ns in missing_imu_timestamps
            )
        ),
        "value_max_absolute_error": float(
            np.max(np.abs(matched_raw_imu_values - matched_converted_imu_values))
        ),
        "initial_100_acceleration_mean_m_s2": imu_initial[:, 3:6].mean(axis=0).tolist(),
        "initial_100_acceleration_norm_mean_m_s2": float(
            np.linalg.norm(imu_initial[:, 3:6], axis=1).mean()
        ),
        "initial_100_gyro_mean_rad_s": imu_initial[:, 0:3].mean(axis=0).tolist(),
    },
    "static_transform_lidar_to_imu": {
        "parent_frame": args.tf_parent,
        "child_frame": args.tf_child,
        "frame_path_child_to_parent": frame_path,
        "matrix": sensor_transform.tolist(),
        "inverse_matrix": np.linalg.inv(sensor_transform).tolist(),
        "determinant": float(np.linalg.det(sensor_transform[:3, :3])),
        "orthogonality_error": float(
            np.linalg.norm(
                sensor_transform[:3, :3].T @ sensor_transform[:3, :3] - np.eye(3)
            )
        ),
    },
    "acceptance_checks": {
        "no_unexpected_message_loss": (
            not missing_lidar_inside_range
            and not missing_lidar_after_range
            and not unexpected_imu_timestamps
            and not any(
                converted_imu_timestamps[0]
                <= timestamp_ns
                <= converted_imu_timestamps[-1]
                for timestamp_ns in missing_imu_timestamps
            )
        ),
        "frame_ids_match_apollo_source": not frame_id_mismatches,
        "sampled_xyz_exact": all(
            entry["xyz_max_absolute_error"] == 0.0 for entry in sample_checks
        ),
        "sampled_intensity_exact": all(
            entry["intensity_max_absolute_error"] == 0.0 for entry in sample_checks
        ),
        "sampled_point_timestamp_error_within_float32_ns": all(
            entry["point_timestamp_max_absolute_error_ns"] <= 8
            for entry in sample_checks
        ),
        "imu_exact": bool(
            not unexpected_imu_timestamps
            and np.max(np.abs(matched_raw_imu_values - matched_converted_imu_values))
            == 0.0
        ),
        "rings_ordered": all(
            ring in ring_elevations
            and ring + 1 in ring_elevations
            and ring_elevations[ring]
            and ring_elevations[ring + 1]
            and np.percentile(ring_elevations[ring], 25)
            > np.percentile(ring_elevations[ring + 1], 75)
            for ring in range(args.scan_lines - 1)
        ),
        "gravity_axis_is_positive_z": bool(imu_initial[:, 3:6].mean(axis=0)[2] > 9.0),
    },
}
if not all(report["acceptance_checks"].values()):
    report["verdict"] = "rejected"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w") as stream:
    yaml.safe_dump(report, stream, sort_keys=False)
print(yaml.safe_dump(report["message_counts"], sort_keys=False))
print(yaml.safe_dump(report["acceptance_checks"], sort_keys=False))
