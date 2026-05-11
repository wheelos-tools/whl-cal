#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging

from lidar2lidar.prepared_dataset import build_prepared_rig_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable raw-LiDAR rig dataset with cached Open3D-readable point clouds."
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="Path to a .record file or a directory containing split record files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2lidar/prepared_rig_dataset",
        help="Directory for the prepared dataset artifacts.",
    )
    parser.add_argument(
        "--lidar-topics",
        nargs="+",
        required=True,
        help="Raw LiDAR topics to keep in the prepared dataset.",
    )
    parser.add_argument(
        "--pose-topic",
        default="/apollo/localization/pose",
        help="Pose topic cached into the dataset.",
    )
    parser.add_argument(
        "--imu-topic",
        default="/apollo/sensor/gnss/imu",
        help="IMU topic cached into the dataset.",
    )
    parser.add_argument(
        "--parent-frame",
        default="imu",
        help="Parent frame used for pose->IMU gravity reconstruction.",
    )
    parser.add_argument(
        "--reference-topic",
        default=None,
        help="Reference LiDAR topic used to define synchronized snapshot anchors.",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=20.0,
        help="Maximum timestamp gap allowed when building synchronized snapshots.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Keep every Nth reference-topic frame before cross-topic synchronization.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Optional cap on synchronized snapshots after frame-stride sampling.",
    )
    parser.add_argument(
        "--export-voxel-size",
        type=float,
        default=0.10,
        help="Optional voxel size applied before writing cached PCD files.",
    )
    args = parser.parse_args()

    dataset_path = build_prepared_rig_dataset(
        record_path=args.record_path,
        output_dir=args.output_dir,
        lidar_topics=args.lidar_topics,
        pose_topic=args.pose_topic,
        imu_topic=args.imu_topic,
        parent_frame=args.parent_frame,
        reference_topic=args.reference_topic,
        sync_threshold_ms=args.sync_threshold_ms,
        frame_stride=args.frame_stride,
        max_snapshots=args.max_snapshots,
        export_voxel_size=args.export_voxel_size,
    )
    logging.info("Prepared rig dataset manifest: %s", dataset_path)


if __name__ == "__main__":
    main()
