"""Extract independent time windows from a canonical dataset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lidar2imu.dataset_io import file_sha256
from lidar2imu.models import CanonicalDataset, ImuBatch, LidarBatch
from lidar2imu.time_axis import replay_hard_offset_ns


def extract_segment(
    dataset: CanonicalDataset,
    *,
    start_ns: int,
    end_ns: int,
    max_lidar_gap_ns: int,
    max_imu_gap_ns: int,
    parent_manifest: Path,
) -> CanonicalDataset:
    """Keep complete scans and IMU samples strictly within [start_ns, end_ns)."""
    dataset.validate()
    if replay_hard_offset_ns(dataset) != 0:
        raise ValueError(
            "prepare-segment requires a shared measurement clock; offset sensor "
            "clocks must be reviewed as a full capture before selecting segments"
        )
    if start_ns >= end_ns:
        raise ValueError("segment start must precede end")
    if max_lidar_gap_ns <= 0 or max_imu_gap_ns <= 0:
        raise ValueError("maximum gaps must be positive")

    lidar = dataset.lidar
    imu = dataset.imu
    first = int(np.searchsorted(lidar.scan_timestamps_ns, start_ns, side="left"))
    stop = int(np.searchsorted(lidar.scan_timestamps_ns, end_ns, side="left"))
    if first == stop:
        raise ValueError("segment contains no LiDAR scans")

    scan_ends = []
    for index in range(first, stop):
        begin, end = lidar.scan_offsets[index : index + 2]
        if begin == end:
            raise ValueError(f"LiDAR scan {index} contains no points")
        scan_ends.append(
            int(lidar.scan_timestamps_ns[index])
            + math.ceil(float(lidar.point_time_s[end - 1]) * 1e9)
        )
    while stop > first and scan_ends[-1] >= end_ns:
        stop -= 1
        scan_ends.pop()
    if stop == first:
        raise ValueError("segment contains no complete LiDAR scans")
    if any(timestamp >= end_ns for timestamp in scan_ends):
        raise ValueError("a LiDAR scan crosses the segment end")

    imu_first = int(np.searchsorted(imu.timestamps_ns, start_ns, side="left"))
    imu_stop = int(np.searchsorted(imu.timestamps_ns, end_ns, side="left"))
    imu_times = imu.timestamps_ns[imu_first:imu_stop]
    scan_times = lidar.scan_timestamps_ns[first:stop]
    if (
        len(imu_times) < 2
        or int(imu_times[0]) > int(scan_times[0])
        or int(imu_times[-1]) < max(scan_ends)
    ):
        raise ValueError("segment IMU does not bracket all retained LiDAR points")
    for name, timestamps, limit in (
        ("LiDAR", scan_times, max_lidar_gap_ns),
        ("IMU", imu_times, max_imu_gap_ns),
    ):
        gaps = np.diff(timestamps)
        if np.any(gaps <= 0):
            raise ValueError(f"{name} timestamps must be strictly increasing")
        if np.any(gaps > limit):
            raise ValueError(f"{name} segment contains a gap exceeding its limit")

    point_first = int(lidar.scan_offsets[first])
    point_stop = int(lidar.scan_offsets[stop])
    metadata = dict(dataset.metadata)
    metadata["segment"] = {
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": file_sha256(parent_manifest),
        "start_ns": start_ns,
        "end_ns_exclusive": end_ns,
        "max_lidar_gap_ns": max_lidar_gap_ns,
        "max_imu_gap_ns": max_imu_gap_ns,
        "source_scan_index_start": first,
        "source_scan_index_end_exclusive": stop,
        "boundary_scans_excluded": int(
            np.searchsorted(lidar.scan_timestamps_ns, end_ns, side="left") - stop
        ),
        "source_imu_index_start": imu_first,
        "source_imu_index_end_exclusive": imu_stop,
    }
    return CanonicalDataset(
        source_type=dataset.source_type,
        source_files=dataset.source_files,
        lidar_topic=dataset.lidar_topic,
        imu_topic=dataset.imu_topic,
        lidar=LidarBatch(
            frame_id=lidar.frame_id,
            scan_timestamps_ns=scan_times,
            scan_offsets=lidar.scan_offsets[first : stop + 1] - point_first,
            xyz=lidar.xyz[point_first:point_stop],
            intensity=lidar.intensity[point_first:point_stop],
            ring=lidar.ring[point_first:point_stop],
            point_time_s=lidar.point_time_s[point_first:point_stop],
            measurement_timestamps_ns=(
                None
                if lidar.measurement_timestamps_ns is None
                else lidar.measurement_timestamps_ns[first:stop]
            ),
            dispatch_timestamps_ns=(
                None
                if lidar.dispatch_timestamps_ns is None
                else lidar.dispatch_timestamps_ns[first:stop]
            ),
        ),
        imu=ImuBatch(
            frame_id=imu.frame_id,
            timestamps_ns=imu_times,
            angular_velocity=imu.angular_velocity[imu_first:imu_stop],
            linear_acceleration=imu.linear_acceleration[imu_first:imu_stop],
            dispatch_timestamps_ns=(
                None
                if imu.dispatch_timestamps_ns is None
                else imu.dispatch_timestamps_ns[imu_first:imu_stop]
            ),
        ),
        transforms=dataset.transforms,
        metadata=metadata,
        schema_version=dataset.schema_version,
    )
