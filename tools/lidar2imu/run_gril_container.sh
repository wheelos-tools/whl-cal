#!/usr/bin/env bash
set -e

IMAGE="whl-cal-lidar2imu:2026.08.30-rc1"

INPUT="${1:-/mnt/synology/weilan/lidar2imu/0827}"
OUTPUT="${2:-$PWD/lidar2imu-output}"

mkdir -p "$OUTPUT"
RUN_MARKER=$(mktemp)
cleanup() {
  rm -f "$RUN_MARKER"
}
trap cleanup EXIT

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=${INPUT},dst=/data,readonly" \
  --mount "type=bind,src=${OUTPUT},dst=/output" \
  "$IMAGE" \
  --input /data \
  --input-type record \
  --output-dir /output &
CONTAINER_PID=$!

STARTED_AT=$(date +%s)
LAST_STAGE=""
while kill -0 "$CONTAINER_PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - STARTED_AT))
  if [[ -f "$OUTPUT/dataset/dataset.yaml" &&
        "$OUTPUT/dataset/dataset.yaml" -nt "$RUN_MARKER" ]]; then
    STAGE="native processing"
  else
    STAGE="reading and decoding Cyber records"
  fi
  if [[ "$STAGE" != "$LAST_STAGE" ]]; then
    printf '\n[%02d:%02d] %s...\n' "$((ELAPSED / 60))" "$((ELAPSED % 60))" "$STAGE"
    LAST_STAGE="$STAGE"
  else
    printf '\r[%02d:%02d] %s' \
      "$((ELAPSED / 60))" "$((ELAPSED % 60))" "$STAGE"
  fi
  sleep 10
done

wait "$CONTAINER_PID"
ELAPSED=$(($(date +%s) - STARTED_AT))
printf '\nFinished in %02d:%02d\n' "$((ELAPSED / 60))" "$((ELAPSED % 60))"
