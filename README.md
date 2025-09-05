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


### Result Example
```
2025-09-05 16:34:18,054 - INFO - Loading point cloud files: PCD/frame_00000.pcd and PCD/frame_00073.pcd
2025-09-05 16:34:18,058 - INFO - --- Starting LiDAR extrinsic calibration ---
2025-09-05 16:34:18,058 - INFO - --- Step 1: Point Cloud Preprocessing ---
2025-09-05 16:34:18,058 - INFO -   -> Cleaning invalid points...
2025-09-05 16:34:18,059 - INFO -      Cleaning invalid points: removed 3115 NaN/Inf points
2025-09-05 16:34:18,061 - INFO -   -> Input points: 25685
2025-09-05 16:34:18,061 - INFO -   -> Voxel downsampling...
2025-09-05 16:34:18,062 - INFO -      After downsampling: 8119 points
2025-09-05 16:34:18,062 - INFO -   -> Removing statistical outliers...
2025-09-05 16:34:18,070 - INFO -      After outlier removal: 7905 points
2025-09-05 16:34:18,070 - INFO -   -> Removing ground points...
2025-09-05 16:34:18,086 - INFO -      After ground removal: 6734 points
2025-09-05 16:34:18,086 - INFO -   -> Removing walls (vertical planes)...
2025-09-05 16:34:18,101 - INFO -     - Removed wall plane 1, points: 1091
2025-09-05 16:34:18,111 - INFO -     - Removed wall plane 2, points: 834
2025-09-05 16:34:18,111 - INFO -   -> Original points: 25685, after preprocessing: 4809
2025-09-05 16:34:18,111 - INFO -   -> Estimating normals...
2025-09-05 16:34:18,116 - INFO -   -> Cleaning invalid points...
2025-09-05 16:34:18,117 - INFO -      Cleaning invalid points: removed 1912 NaN/Inf points
2025-09-05 16:34:18,119 - INFO -   -> Input points: 26888
2025-09-05 16:34:18,119 - INFO -   -> Voxel downsampling...
2025-09-05 16:34:18,121 - INFO -      After downsampling: 9951 points
2025-09-05 16:34:18,121 - INFO -   -> Removing statistical outliers...
2025-09-05 16:34:18,133 - INFO -      After outlier removal: 9709 points
2025-09-05 16:34:18,133 - INFO -   -> Removing ground points...
2025-09-05 16:34:18,163 - INFO -      After ground removal: 7796 points
2025-09-05 16:34:18,166 - INFO -   -> Removing walls (vertical planes)...
2025-09-05 16:34:18,180 - INFO -   -> Original points: 26888, after preprocessing: 7796
2025-09-05 16:34:18,181 - INFO -   -> Estimating normals...
2025-09-05 16:34:18,195 - INFO - --- Step 2: FPFH Feature Extraction ---
2025-09-05 16:34:18,195 - INFO -   -> Computing FPFH features...
2025-09-05 16:34:18,215 - INFO -   -> Computing FPFH features...
2025-09-05 16:34:18,249 - INFO - --- Step 3: Coarse Registration (FPFH + RANSAC) ---
2025-09-05 16:34:18,249 - INFO -   -> Performing coarse registration (FPFH + RANSAC)...
2025-09-05 16:34:25,886 - INFO - 
--- Coarse Calibration Metrics ---
2025-09-05 16:34:25,887 - INFO - Coarse registration matched points: 4503
2025-09-05 16:34:25,887 - INFO -   - Fitness (overlap): 0.9364
2025-09-05 16:34:25,888 - INFO -   - Inlier RMSE: 0.1723
2025-09-05 16:34:25,888 - INFO -   - Transformation Matrix:
[[ 0.26736072 -0.96215448 -0.05269738 -0.1734026 ]
 [ 0.96251824  0.26924832 -0.03261859  0.7925205 ]
 [ 0.04557281 -0.04200126  0.99807766 -0.06592043]
 [ 0.          0.          0.          1.        ]]
2025-09-05 16:34:25,890 - INFO -   - Rotation (quaternion wxyz): [np.float64(0.7960349709051228), np.float64(-0.002946687811927389), np.float64(-0.030862396441649793), np.float64(0.6044560816245443)]
2025-09-05 16:34:25,890 - INFO -   - Translation (xyz): [-0.1734026   0.7925205  -0.06592043]
2025-09-05 16:34:25,890 - INFO - --- Step 4: Fine Registration (Iterative GICP) ---
2025-09-05 16:34:25,890 - INFO -   -> Iteration 1: Max correspondence distance 1.000 m
2025-09-05 16:34:25,890 - INFO -   -> Using point-to-plane ICP
2025-09-05 16:34:25,905 - INFO -   -> Iteration 2: Max correspondence distance 0.500 m
2025-09-05 16:34:25,905 - INFO -   -> Using point-to-plane ICP
2025-09-05 16:34:25,991 - INFO -   -> Iteration 3: Max correspondence distance 0.100 m
2025-09-05 16:34:25,995 - INFO -   -> Using point-to-plane ICP
2025-09-05 16:34:26,055 - INFO -   -> Iteration 4: Max correspondence distance 0.050 m
2025-09-05 16:34:26,056 - INFO -   -> Using point-to-plane ICP
2025-09-05 16:34:26,113 - INFO -   -> Iteration 5: Max correspondence distance 0.020 m
2025-09-05 16:34:26,113 - INFO -   -> Using point-to-plane ICP
2025-09-05 16:34:26,270 - INFO - 
--- Final Calibration Metrics ---
2025-09-05 16:34:26,296 - INFO - Fine registration matched points (within 0.40 m): 1533
2025-09-05 16:34:26,296 - INFO -   - Fitness (overlap): 0.1485
2025-09-05 16:34:26,296 - INFO -   - Inlier RMSE: 0.0146
2025-09-05 16:34:26,296 - INFO -   - Transformation Matrix:
[[ 0.29707988 -0.95485166 -0.00135982 -0.11084946]
 [ 0.95467624  0.29705207 -0.0187972   0.56048105]
 [ 0.01835248  0.00428608  0.99982239 -0.01395672]
 [ 0.          0.          0.          1.        ]]
2025-09-05 16:34:26,297 - INFO -   - Rotation (quaternion wxyz): [np.float64(0.8052878901185664), np.float64(0.0071661582212755445), np.float64(-0.006119644451552244), np.float64(0.5928090840705141)]
2025-09-05 16:34:26,297 - INFO -   - Translation (xyz): [-0.11084946  0.56048105 -0.01395672]
2025-09-05 16:34:26,297 - INFO - Visualizing coarse registration...
2025-09-05 16:34:31,883 - INFO - Visualizing fine registration...
2025-09-05 16:34:35,931 - INFO - 
--- Final calibration result ---
2025-09-05 16:34:35,932 - INFO - Computed final extrinsic matrix:
[[ 0.29708  -0.954852 -0.00136  -0.110849]
 [ 0.954676  0.297052 -0.018797  0.560481]
 [ 0.018352  0.004286  0.999822 -0.013957]
 [ 0.        0.        0.        1.      ]]
2025-09-05 16:34:35,932 - INFO - Fitness: 0.148472
2025-09-05 16:34:35,932 - INFO - Inlier RMSE: 0.014623
2025-09-05 16:34:35,932 - INFO - Final extrinsic matrix saved to result.json

```

### Explanation

- Fitness: ratio of valid correspondences to source points  
- Inlier RMSE: root mean square error of inlier correspondences (sensitive to threshold)  
- Extrinsic matrix: the computed rigid transformation matrix (rotation + translation)  

