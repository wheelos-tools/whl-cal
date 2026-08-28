#!/usr/bin/env bash
set -euo pipefail

BAG_PATH=${1:?bag path under /data is required}
RUN_COUNT=${2:-2}

source /opt/ros/noetic/setup.bash
source /root/livox_ws/devel/setup.bash
source /root/catkin_ws/devel/setup.bash

roscore >/data/roscore.log 2>&1 &
ROSCORE_PID=$!
trap 'kill "$ROSCORE_PID" 2>/dev/null || true' EXIT
sleep 3

for RUN in $(seq 1 "$RUN_COUNT"); do
    RUN_DIR="/data/run_${RUN}"
    mkdir -p "$RUN_DIR/log"
    roslaunch gril_calib velodyne.launch rviz:=false \
        >"$RUN_DIR/gril.log" 2>&1 &
    GRIL_PID=$!
    sleep 5
    rosbag play "$BAG_PATH" >"$RUN_DIR/play.log" 2>&1
    wait "$GRIL_PID"
    cp /root/catkin_ws/src/gril_calib/result/GRIL_Calib_result.txt \
        "$RUN_DIR/GRIL_Calib_result.txt"
    cp /root/catkin_ws/src/gril_calib/Log/*.txt "$RUN_DIR/log/"
    cp /data/lidar_trajectory.txt "$RUN_DIR/lidar_trajectory.txt"
    sleep 2
done
