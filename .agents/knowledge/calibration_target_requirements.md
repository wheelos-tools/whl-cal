# Calibration input requirements

| Workflow | Required calibration evidence |
| --- | --- |
| Camera intrinsic | Physical planar target observations with stable camera mode and broad image coverage |
| LiDAR-to-camera | Paired images and point clouds, camera intrinsics/distortion, and a checkerboard visible in both sensors |
| LiDAR-to-LiDAR | Shared scene geometry, adequate overlap, and sufficient scene coverage; no board is required by the current baseline |
| LiDAR-to-IMU | Correct timestamp/frame/point-time contract and sufficiently observable motion; no board is required by GRIL |

These are input requirements, not guarantees of accuracy. Use the module
quickstarts for collection procedures and acceptance criteria. The longer
evidence review and external references are retained in
[research notes](../notes/calibration_target_requirements.md).
