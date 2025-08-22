需求
配准雷达点云数据，获取外参矩阵（或四元数及平移向量）
QuickStart
https://github.com/wheelos-tools/whl-cal
分支：dev-optimize
安装依赖
pip install -r requirements.txt
运行
数据准备
该文档使用的数据源是使用apollo中的cyber_record采集的数据
从录制的数据包提取点云pcd
python3 record2pcd.py record_file output_dir
# record_file 更换为录制包的路径
# output_dir 更换为用于保存输出的文件夹路径
如果需要修改需要保存的点云数据Topic，请前往代码中自行修改:
# List of target topics to export
TARGET_TOPICS = [
    "/apollo/sensor/vanjeelidar/left_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_back/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/left_back/PointCloud2",
]
标定外参
单次配准 ( cli.py )
  执行命令
python3 cli.py --source-pcd your_source_pcd --target-pcd your_target_pcd [--output-json Path] [--visualize]
注：[ ] 表示可选参数

---
- --source-pcd：原始点云路径
- --target-pcd：目标点云路径
- --output-json：结果输出 json 文件路径，默认路径为当前目录的calibration_result.json
- --visualize：可视化配准点云数据，绿色为原始点云经配准转换后的点云显示，蓝色为目标点云显示，红色连线为两个点云之间的配准点显示
结果说明
2025-08-22 12:02:30,343 - INFO - Loading point cloud files: 00000/left_back_000001.pcd and 00000/right_back_000001.pcd
2025-08-22 12:02:30,379 - INFO - --- Starting LiDAR extrinsic calibration ---
2025-08-22 12:02:30,380 - INFO - --- Step 1: Point Cloud Preprocessing ---
2025-08-22 12:02:30,380 - INFO -   -> Voxel downsampling...
2025-08-22 12:02:30,399 - INFO -   -> Removing statistical outliers...
2025-08-22 12:02:30,457 - INFO -   -> Removing ground points...
2025-08-22 12:02:30,542 - INFO -   -> Removing walls (vertical planes)...
2025-08-22 12:02:30,605 - INFO -     - Removed wall plane 1, points: 3808
2025-08-22 12:02:30,650 - INFO -   -> Original points: 57600, after preprocessing: 15346
2025-08-22 12:02:30,650 - INFO -   -> Estimating normals...
2025-08-22 12:02:30,677 - INFO -   -> Voxel downsampling...
2025-08-22 12:02:30,690 - INFO -   -> Removing statistical outliers...
2025-08-22 12:02:30,719 - INFO -   -> Removing ground points...
2025-08-22 12:02:30,793 - INFO -   -> Removing walls (vertical planes)...
2025-08-22 12:02:30,841 - INFO -     - Removed wall plane 1, points: 2463
2025-08-22 12:02:30,891 - INFO -   -> Original points: 57600, after preprocessing: 16996
2025-08-22 12:02:30,896 - INFO -   -> Estimating normals...
2025-08-22 12:02:30,923 - INFO - --- Step 2: FPFH Feature Extraction ---
2025-08-22 12:02:30,923 - INFO -   -> Computing FPFH features...
2025-08-22 12:02:31,014 - INFO -   -> Computing FPFH features...
2025-08-22 12:02:31,116 - INFO - --- Step 3: Coarse Registration (FPFH + RANSAC) ---
2025-08-22 12:02:31,116 - INFO -   -> Performing coarse registration (FPFH + RANSAC)...
2025-08-22 12:02:40,377 - INFO - 
--- Coarse Calibration Metrics ---
2025-08-22 12:02:40,378 - INFO - Coarse registration matched points: 7465
2025-08-22 12:02:40,378 - INFO -   - Fitness (overlap): 0.4864
2025-08-22 12:02:40,378 - INFO -   - Inlier RMSE: 0.1636
2025-08-22 12:02:40,378 - INFO -   - Transformation Matrix:
[[-0.02424564  0.99933014 -0.02741203 -1.51444568]
 [-0.99959637 -0.02463999 -0.01414116 -1.46223593]
 [-0.01480712  0.02705811  0.99952419  0.06987993]
 [ 0.          0.          0.          1.        ]]
2025-08-22 12:02:40,379 - INFO -   - Rotation (quaternion wxyz): [np.float64(-0.6983263131853024), np.float64(-0.01474928773319429), np.float64(0.00451254589742375), np.float64(0.7156133423531944)]
2025-08-22 12:02:40,380 - INFO -   - Translation (xyz): [-1.51444568 -1.46223593  0.06987993]
2025-08-22 12:02:40,380 - INFO - --- Step 4: Fine Registration (Iterative GICP) ---
2025-08-22 12:02:40,380 - INFO -   -> Iteration 1: Max correspondence distance 1.000 m
2025-08-22 12:02:40,587 - INFO -   -> Iteration 2: Max correspondence distance 0.500 m
2025-08-22 12:02:40,829 - INFO -   -> Iteration 3: Max correspondence distance 0.100 m
2025-08-22 12:02:40,914 - INFO -   -> Iteration 4: Max correspondence distance 0.050 m
2025-08-22 12:02:41,026 - INFO -   -> Iteration 5: Max correspondence distance 0.020 m
2025-08-22 12:02:41,131 - INFO - 
--- Final Calibration Metrics ---
2025-08-22 12:02:41,249 - INFO - Fine registration matched points (within 0.40 m): 2097
2025-08-22 12:02:41,249 - INFO -   - Fitness (overlap): 0.0430
2025-08-22 12:02:41,249 - INFO -   - Inlier RMSE: 0.0144
2025-08-22 12:02:41,249 - INFO -   - Transformation Matrix:
[[-0.02835223  0.99915668 -0.02969971 -1.42831515]
 [-0.99958566 -0.02848698 -0.00412371 -1.43925404]
 [-0.00496629  0.02957049  0.99955036 -0.02661979]
 [ 0.          0.          0.          1.        ]]
2025-08-22 12:02:41,250 - INFO -   - Rotation (quaternion wxyz): [np.float64(-0.6969058685970488), np.float64(-0.012087070849316194), np.float64(0.008872585340285394), np.float64(0.7170058509263757)]
2025-08-22 12:02:41,250 - INFO -   - Translation (xyz): [-1.42831515 -1.43925404 -0.02661979]
2025-08-22 12:02:41,250 - INFO - Visualizing coarse registration...
2025-08-22 13:33:22,712 - INFO - Visualizing fine registration...
2025-08-22 13:33:29,643 - INFO - 
--- Final calibration result ---
2025-08-22 13:33:29,643 - INFO - Computed final extrinsic matrix:
[[-0.028352  0.999157 -0.0297   -1.428315]
 [-0.999586 -0.028487 -0.004124 -1.439254]
 [-0.004966  0.02957   0.99955  -0.02662 ]
 [ 0.        0.        0.        1.      ]]
2025-08-22 13:33:29,643 - INFO - Fitness: 0.043008
2025-08-22 13:33:29,644 - INFO - Inlier RMSE: 0.014413
2025-08-22 13:33:29,644 - INFO - Final extrinsic matrix saved to calibration_result.json
  最后输出类似结果，Fitness表示有效匹配/源点数 ，Inlier RMSE表示均方根误差 （受阈值影响较大），extrinsic matrix表示外参变换矩阵 。
批量配准 ( cli_batch.py )
功能说明
  本代码仅适用于四个激光雷达点云的配准
输入说明
  需要多帧点云数据，且包含以下名称的文件夹
lidar_names = ["left_back", "right_back", "right_front", "left_front"]
[图片]
每个文件夹包含各自激光雷达采集的多帧点云pcd文件
输出说明
会在输出目录生成以下文件
final.txt —— 包含左后到右后、右后到右前、右前到左前、左前到左后的用于配准激光雷达点云的外参矩阵
left_back2right_back.yaml —— 左后配准到右后激光雷达的旋转四元数与平移向量
left_front2left_back.yaml —— 左前配准到左后激光雷达的旋转四元数与平移向量
right_back2right_front.yaml —— 右后配准到右前激光雷达的旋转四元数与平移向量
right_front2left_front.yaml —— 右前配准到左前激光雷达的旋转四元数与平移向量
执行命令
python3 cli_batch.py --parent_dir PATH [--output_dir PATH] [--visualize] [--max_frames NUMBER]
注：[ ] 表示可选参数

---
- --parent_dir：在输入说明中四个文件夹所在的文件夹路径
- --output_dir：用于保存输出结果的文件夹路径，默认为当前目录的output文件夹中，如果没有会新建一个
- --visualize：可视化配准结果，与单次配准的功能一致，但是不推荐在这里使用，批量处理的进程速度会受可视化影响，只有关闭可视化窗口批量配准程序才能继续进行
- --max_frames：用于配准的最大帧数，默认为10帧数据，然后将结果取平均值
融合点云
执行命令
python3 fusion.py --lf-pcd PATH --rf-pcd PATH --rb-pcd PATH --lb-pcd PATH --tf-dir PATH [--use-pclview]
注：[ ] 表示可选参数

---
- --lf-pcd：左前激光雷达点云路径
- --rf-pcd：右前激光雷达点云路径
- --rb-pcd：右后激光雷达点云路径
- --lb-pcd：左后激光雷达点云路径
- --tf-dir：执行批量配准代码后生成的输出目标文件夹路径，即该路径需要有 left_back2right_back.yaml，right_back2right_front.yaml，right_front2left_front.yaml 文件，本程序将所有点云数据配准到左前激光雷达坐标系上，因此只需要有以上三个文件即可
- --use-pclview：使用 pclview 进行可视化显示，如果不输入此参数默认使用 open3d 进行可视化显示

如果选择 pclview 并执行后报错，检查系统是否安装 pclview ，如果没有需要在终端执行以下命令：
sudo apt update && sudo apt install -y pcl-tools
使用以下命令检查是否安装成功
which pcl_viewer
如果成功安装，会出现以下结果
/usr/bin/pcl_viewer

