#!/usr/bin/env python3
"""Extract PointCloud2 messages from Apollo record files into PCD files."""

from __future__ import annotations

import datetime
import os
import sys

import click
from cyber_record.record import Record
from record_msg.parser import PointCloudParser


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.record_utils import discover_record_files


def should_extract_channel(channel: str, exact_topics: list[str], prefix_patterns: list[str]) -> bool:
    return channel in exact_topics or any(channel.startswith(prefix) for prefix in prefix_patterns)


def output_file_stem(channel: str, timestamp_ns: int) -> tuple[str, str]:
    sensor_name = channel.split("/")[-2]
    timestamp = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
    file_stem = timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    return sensor_name, file_stem


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to a record file or a directory containing record files.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory for extracted PCD files.",
)
@click.option(
    "--channel",
    "channels",
    "-c",
    type=str,
    multiple=True,
    default=(
        "/apollo/sensor/vanjeelidar/left_front/PointCloud2",
        "/apollo/sensor/vanjeelidar/left_back/PointCloud2",
        "/apollo/sensor/vanjeelidar/right_front/PointCloud2",
        "/apollo/sensor/vanjeelidar/right_back/PointCloud2",
        "/apollo/sensor/lidar/fusion/PointCloud2",
    ),
    help="Exact topic or topic-prefix ending with '/'. Repeat to extract multiple channels.",
)
def main(input_dir: str, output_dir: str, channels: tuple[str, ...]) -> None:
    record_files = discover_record_files(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    exact_topics = [channel for channel in channels if not channel.endswith("/")]
    prefix_patterns = [channel for channel in channels if channel.endswith("/")]

    parser = PointCloudParser(output_dir, True, ".pcd")
    for record_file in record_files:
        with Record(record_file) as record:
            message_iter = record.read_messages(topics=tuple(exact_topics)) if not prefix_patterns else record.read_messages()
            for channel, message, timestamp_ns in message_iter:
                if not should_extract_channel(channel, exact_topics, prefix_patterns):
                    continue
                if not hasattr(message, "point"):
                    continue

                sensor_name, file_stem = output_file_stem(channel, timestamp_ns)
                os.makedirs(os.path.join(output_dir, sensor_name), exist_ok=True)
                parser.parse(message, file_name=f"{sensor_name}/{file_stem}", mode="ascii")


if __name__ == "__main__":
    main()
