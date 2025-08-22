#!/usr/bin/env python3
"""Export Apollo record PointCloud2 messages to PCD files."""

import os
import logging
import argparse
import numpy as np
import open3d as o3d
from cyber_record.record import Record

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# List of target topics to export
TARGET_TOPICS = [
    "/apollo/sensor/vanjeelidar/left_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_back/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/left_back/PointCloud2",
]


def save_pointcloud_to_pcd(msg, output_base_dir: str, topic_name: str, frame_idx: int) -> None:
    """Convert Apollo PointCloud2 message to PCD and save to disk.

    Args:
        msg: Apollo PointCloud2 message.
        output_base_dir: Root directory to save output PCDs.
        topic_name: Full topic name of the message.
        frame_idx: Frame index number.
    """
    points = np.array([[p.x, p.y, p.z] for p in msg.point], dtype=np.float32)
    if points.size == 0:
        logging.warning("Frame %d of %s is empty, skipping.", frame_idx, topic_name)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Map intensity to grayscale color if available
    intensities = np.array([p.intensity for p in msg.point], dtype=np.float32)
    if intensities.size > 0 and intensities.max() > 0:
        norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
        colors = np.stack([norm_intensities] * 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    topic_short = topic_name.split("/")[-2]  # e.g., left_front
    output_dir = os.path.join(output_base_dir, topic_short)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{topic_short}_{frame_idx:06d}.pcd")
    o3d.io.write_point_cloud(filename, pcd)
    logging.info("Saved %s (%d points)", filename, len(points))


def export_record_to_pcd(record_file: str, output_base_dir: str) -> None:
    """Export all target topics from an Apollo record to PCD files.

    Args:
        record_file: Path to the .record file.
        output_base_dir: Directory to save PCD files.
    """
    record = Record(record_file)
    frame_counter = {topic: 0 for topic in TARGET_TOPICS}

    for topic, msg, _ in record.read_messages():
        if topic in TARGET_TOPICS:
            frame_counter[topic] += 1
            save_pointcloud_to_pcd(msg, output_base_dir, topic, frame_counter[topic])


def main():
    """Parse arguments and start the export process."""
    parser = argparse.ArgumentParser(description="Export Apollo PointCloud2 messages to PCD files.")
    parser.add_argument("record_file", type=str, help="Path to the input .record file.")
    parser.add_argument("output_dir", type=str, help="Directory to save PCD files.")
    args = parser.parse_args()

    if not os.path.isfile(args.record_file):
        logging.error("Input record file does not exist: %s", args.record_file)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    export_record_to_pcd(args.record_file, args.output_dir)
    logging.info("Export completed successfully.")


if __name__ == "__main__":
    main()
