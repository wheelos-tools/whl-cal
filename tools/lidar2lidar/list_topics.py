#!/usr/bin/env python3
"""List topics in an Apollo record file or directory."""

from __future__ import annotations

import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.record_utils import discover_record_files, list_topics


def main() -> None:
    parser = argparse.ArgumentParser(description="List topics in an Apollo .record file or split-record directory.")
    parser.add_argument("record_path", help="Path to a .record file or a directory containing split record files.")
    args = parser.parse_args()

    record_files = discover_record_files(args.record_path)
    counts = list_topics(record_files)
    for topic, count in counts.items():
        print(f"{topic} {count}")


if __name__ == "__main__":
    main()
