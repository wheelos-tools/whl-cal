# Requirement

Register LiDAR point cloud data to obtain the extrinsic matrix.  

# Quick Start

Repository: https://github.com/wheelos-tools/whl-cal  

## Install dependencies

pip install -e .

## Extrinsic Calibration

### Single registration (cli.py)

Run command:

python3 cli.py --source-pcd your_source_pcd --target-pcd your_target_pcd [--output-json Path] [--visualize]

Note: [ ] means optional parameters

- --source-pcd: source point cloud file path
- --target-pcd: target point cloud file path
- --output-json: output result JSON file path, default is calibration_result.json in the current directory
- --visualize: visualize the registration result
  - Green: source point cloud after registration
  - Blue: target point cloud
  - Red lines: correspondences between the two point clouds

---

### Result Example

2025-08-22 12:02:30,343 - INFO - Loading point cloud files: 00000/left_back_000001.pcd and 00000/right_back_000001.pcd
2025-08-22 12:02:30,379 - INFO - --- Starting LiDAR extrinsic calibration ---
2025-08-22 12:02:30,380 - INFO - --- Step 1: Point Cloud Preprocessing ---
...
2025-08-22 12:02:41,249 - INFO - 
--- Final Calibration Metrics ---
2025-08-22 12:02:41,249 - INFO - Fine registration matched points (within 0.40 m): 2097
2025-08-22 12:02:41,249 - INFO -   - Fitness (overlap): 0.0430
2025-08-22 12:02:41,249 - INFO -   - Inlier RMSE: 0.0144
2025-08-22 12:02:41,249 - INFO -   - Transformation Matrix:
[[-0.02835223  0.99915668 -0.02969971 -1.42831515]
 [-0.99958566 -0.02848698 -0.00412371 -1.43925404]
 [-0.00496629  0.02957049  0.99955036 -0.02661979]
 [ 0.          0.          0.          1.        ]]
2025-08-22 12:02:41,250 - INFO -   - Rotation (quaternion wxyz): [-0.6969058685970488, -0.012087070849316194, 0.008872585340285394, 0.7170058509263757]
2025-08-22 12:02:41,250 - INFO -   - Translation (xyz): [-1.42831515 -1.43925404 -0.02661979]
...
--- Final calibration result ---
2025-08-22 13:33:29,643 - INFO - Computed final extrinsic matrix:
[[-0.028352  0.999157 -0.0297   -1.428315]
 [-0.999586 -0.028487 -0.004124 -1.439254]
 [-0.004966  0.02957   0.99955  -0.02662 ]
 [ 0.        0.        0.        1.      ]]
2025-08-22 13:33:29,643 - INFO - Fitness: 0.043008
2025-08-22 13:33:29,644 - INFO - Inlier RMSE: 0.014413
2025-08-22 13:33:29,644 - INFO - Final extrinsic matrix saved to calibration_result.json

---

### Explanation

- Fitness: ratio of valid correspondences to source points  
- Inlier RMSE: root mean square error of inlier correspondences (sensitive to threshold)  
- Extrinsic matrix: the computed rigid transformation matrix (rotation + translation)  

