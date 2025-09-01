# Requirement

Register LiDAR point cloud data to obtain the extrinsic matrix.  

# Quick Start

Repository: https://github.com/wheelos-tools/whl-cal  

## Install dependencies

pip install -e .

## Extrinsic Calibration

### Single registration (cli.py)

Run command:

python3 cli.py --source-pcd your_source_pcd --target-pcd your_target_pcd [--output-json PATH] [--visualize][--voxel-size FLOAT][--max-height FLOAT][--method INT]

Note: [ ] means optional parameters

- --source-pcd: source point cloud file path
- --target-pcd: target point cloud file path
- --output-json: output result JSON file path, default is calibration_result.json in the current directory
- --visualize: visualize the registration result
  - Green: source point cloud after registration
  - Blue: target point cloud
  - Red lines: correspondences between the two point clouds
- --voxel-size: set the downsampling parameters for the coarse registration process, default is 0.04
- --max-height: set whether to filter out points above a certain height (in meters) during the coarse registration process, default is None
- --method: set the registration method used during the fine registration process, default is 1
  - 1: point to plane
  - 2: GICP
  - 3: point to point

---

### Result Example

2025-08-27 16:58:53,079 - INFO - Loading point cloud files: pcd_data/left/fused_cloud.pcd and pcd_data/right/fused_cloud.pcd
2025-08-27 16:58:53,082 - INFO - --- Starting LiDAR extrinsic calibration ---
2025-08-27 16:58:53,082 - INFO - --- Step 1: Point Cloud Preprocessing ---
2025-08-27 16:58:53,082 - INFO -   -> Voxel downsampling...
2025-08-27 16:58:53,084 - INFO -   -> Removing statistical outliers...
2025-08-27 16:58:53,091 - INFO -   -> Removing ground points...
2025-08-27 16:58:53,104 - INFO -   -> Removing walls (vertical planes)...
2025-08-27 16:58:53,117 - INFO -     - Removed wall plane 1, points: 798
2025-08-27 16:58:53,123 - INFO -   -> Original points: 9358, after preprocessing: 6652
2025-08-27 16:58:53,124 - INFO -   -> Estimating normals...
2025-08-27 16:58:53,127 - INFO -   -> Voxel downsampling...
2025-08-27 16:58:53,128 - INFO -   -> Removing statistical outliers...
2025-08-27 16:58:53,132 - INFO -   -> Removing ground points...
2025-08-27 16:58:53,141 - INFO -   -> Removing walls (vertical planes)...
2025-08-27 16:58:53,153 - INFO -     - Removed wall plane 1, points: 635
2025-08-27 16:58:53,163 - INFO -     - Removed wall plane 2, points: 677
2025-08-27 16:58:53,163 - INFO -   -> Original points: 11207, after preprocessing: 7565
2025-08-27 16:58:53,163 - INFO -   -> Estimating normals...
2025-08-27 16:58:53,166 - INFO - --- Step 2: FPFH Feature Extraction ---
2025-08-27 16:58:53,166 - INFO -   -> Computing FPFH features...
2025-08-27 16:58:53,181 - INFO -   -> Computing FPFH features...
2025-08-27 16:58:53,199 - INFO - --- Step 3: Coarse Registration (FPFH + RANSAC) ---
2025-08-27 16:58:53,199 - INFO -   -> Performing coarse registration (FPFH + RANSAC)...
2025-08-27 16:58:56,589 - INFO - 
--- Coarse Calibration Metrics ---
2025-08-27 16:58:56,589 - INFO - Coarse registration matched points: 4197
2025-08-27 16:58:56,590 - INFO -   - Fitness (overlap): 0.6309
2025-08-27 16:58:56,590 - INFO -   - Inlier RMSE: 0.1987
2025-08-27 16:58:56,590 - INFO -   - Transformation Matrix:
[[ 0.98920706 -0.14544096  0.0177855  -0.07522706]
 [ 0.14470428  0.9887684   0.03738621  1.67708641]
 [-0.02302323 -0.03440907  0.99914261  0.13040459]
 [ 0.          0.          0.          1.        ]]
2025-08-27 16:58:56,590 - INFO -   - Rotation (quaternion wxyz): [np.float64(0.9971356552784617), np.float64(-0.01800037838238779), np.float64(0.010231488846756098), np.float64(0.0727446766881422)]
2025-08-27 16:58:56,590 - INFO -   - Translation (xyz): [-0.07522706  1.67708641  0.13040459]
2025-08-27 16:58:56,590 - INFO - --- Step 4: Fine Registration (Iterative GICP) ---
2025-08-27 16:58:56,591 - INFO -   -> Iteration 1: Max correspondence distance 1.000 m
2025-08-27 16:58:56,591 - INFO -   -> Using point-to-plane ICP
2025-08-27 16:58:56,613 - INFO -   -> Iteration 2: Max correspondence distance 0.500 m
2025-08-27 16:58:56,613 - INFO -   -> Using point-to-plane ICP
2025-08-27 16:58:56,670 - INFO -   -> Iteration 3: Max correspondence distance 0.100 m
2025-08-27 16:58:56,671 - INFO -   -> Using point-to-plane ICP
2025-08-27 16:58:56,708 - INFO -   -> Iteration 4: Max correspondence distance 0.050 m
2025-08-27 16:58:56,708 - INFO -   -> Using point-to-plane ICP
2025-08-27 16:58:56,788 - INFO -   -> Iteration 5: Max correspondence distance 0.020 m
2025-08-27 16:58:56,789 - INFO -   -> Using point-to-plane ICP
2025-08-27 16:58:56,796 - INFO - 
--- Final Calibration Metrics ---
2025-08-27 16:58:56,825 - INFO - Fine registration matched points (within 0.40 m): 1161
2025-08-27 16:58:56,825 - INFO -   - Fitness (overlap): 0.0544
2025-08-27 16:58:56,825 - INFO -   - Inlier RMSE: 0.0139
2025-08-27 16:58:56,825 - INFO -   - Transformation Matrix:
[[ 0.98242318 -0.18639917 -0.01000242 -0.1520441 ]
 [ 0.18656456  0.98224956  0.01948011  1.47645053]
 [ 0.0061938  -0.02100381  0.99976021  0.13158661]
 [ 0.          0.          0.          1.        ]]
2025-08-27 16:58:56,825 - INFO -   - Rotation (quaternion wxyz): [np.float64(0.9955441909496608), np.float64(-0.010166280200628958), np.float64(-0.0040671771908917555), np.float64(0.0936582547507987)]
2025-08-27 16:58:56,825 - INFO -   - Translation (xyz): [-0.1520441   1.47645053  0.13158661]
2025-08-27 16:58:56,825 - INFO - Visualizing coarse registration...
2025-08-27 16:58:57,781 - INFO - Visualizing fine registration...
2025-08-27 16:58:58,601 - INFO - 
--- Final calibration result ---
2025-08-27 16:58:58,601 - INFO - Computed final extrinsic matrix:
[[ 0.982423 -0.186399 -0.010002 -0.152044]
 [ 0.186565  0.98225   0.01948   1.476451]
 [ 0.006194 -0.021004  0.99976   0.131587]
 [ 0.        0.        0.        1.      ]]
2025-08-27 16:58:58,601 - INFO - Fitness: 0.054420
2025-08-27 16:58:58,601 - INFO - Inlier RMSE: 0.013876
2025-08-27 16:58:58,602 - INFO - Final extrinsic matrix saved to calibration_result.json

---

### Explanation

- Fitness: ratio of valid correspondences to source points  
- Inlier RMSE: root mean square error of inlier correspondences (sensitive to threshold)  
- Extrinsic matrix: the computed rigid transformation matrix (rotation + translation)  
