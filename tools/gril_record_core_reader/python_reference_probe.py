#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from pathlib import Path

from lidar2lidar.record_adapter import Record

TOPICS = (
    "/apollo/sensor/gnss/imu",
    "/apollo/sensor/vanjeelidar/up/PointCloud2",
)
MASK = (1 << 64) - 1


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python_reference_probe.py RECORD [RECORD ...]", file=sys.stderr)
        return 2

    stats = {
        topic: {
            "count": 0,
            "min_timestamp": MASK,
            "max_timestamp": 0,
            "hash": 14695981039346656037,
        }
        for topic in TOPICS
    }
    total_messages = 0
    started = time.perf_counter()

    for filename in map(Path, sys.argv[1:]):
        with Record(str(filename)) as record:
            if record._backend_name != "cyber_record":
                raise RuntimeError(
                    "This comparison probe requires the cyber_record backend"
                )
            reader = record._impl._py_reader
            total_messages += sum(
                channel.message_number for channel in reader.channels.values()
            )
            for topic, payload, _, timestamp in record.read_raw_messages(topics=TOPICS):
                row = stats[topic]
                timestamp = int(timestamp)
                row["count"] += 1
                row["min_timestamp"] = min(row["min_timestamp"], timestamp)
                row["max_timestamp"] = max(row["max_timestamp"], timestamp)
                data = (
                    timestamp.to_bytes(8, "little") + topic.encode() + b"\0" + payload
                )
                for byte in data:
                    row["hash"] = ((row["hash"] ^ byte) * 1099511628211) & MASK

    print(f"elapsed_s={time.perf_counter() - started:.3f}")
    print(f"total_messages={total_messages}")
    for topic, row in stats.items():
        print(
            f"topic={topic} count={row['count']} "
            f"min_timestamp_ns={row['min_timestamp']} "
            f"max_timestamp_ns={row['max_timestamp']} "
            f"hash={row['hash']:016x}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
