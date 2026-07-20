#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/lidar2lidar/native/pcl_gicp_align.cpp"
OUTPUT="$ROOT/lidar2lidar/bin/pcl_gicp_align"

mkdir -p "$(dirname "$OUTPUT")"
g++ -O3 -std=c++17 "$SOURCE" -o "$OUTPUT" $(pkg-config --cflags --libs pcl_common pcl_io pcl_registration)
chmod +x "$OUTPUT"
echo "Built $OUTPUT"
