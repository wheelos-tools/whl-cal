import argparse
import logging
import numpy as np
import open3d as o3d
import lidar_calibrator as lc
import copy
import sys

# Configure logging for the command-line tool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_point_clouds(source_path, target_path):
    """
    Load point clouds from specified file paths.
    """
    logging.info(f"加载点云文件: {source_path} 和 {target_path}")
    source_cloud = o3d.io.read_point_cloud(source_path)
    target_cloud = o3d.io.read_point_cloud(target_path)

    if not source_cloud.has_points() or not target_cloud.has_points():
        logging.error("无法加载点云文件或点云为空。")
        sys.exit(1)

    return source_cloud, target_cloud

def main():
    parser = argparse.ArgumentParser(description="LiDAR to LiDAR Extrinsic Calibration Tool.")
    parser.add_argument("--source-pcd", type=str, required=True, help="Path to the source LiDAR point cloud file (e.g., lidar1.pcd).")
    parser.add_argument("--target-pcd", type=str, required=True, help="Path to the target LiDAR point cloud file (e.g., lidar2.pcd).")
    parser.add_argument("--output-json", type=str, default="calibration_result.json", help="Path to save the final extrinsic matrix.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the registration results.")

    args = parser.parse_args()

    # 加载点云
    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)

    # 执行校准
    logging.info("--- 开始 LiDAR 外参校准 ---")
    final_extrinsic_transform, coarse_transform, reg_result = lc.calibrate_lidar_extrinsic(
        source_cloud, target_cloud
    )

    if final_extrinsic_transform is None:
        logging.error("校准失败。")
        return

    # 打印和保存结果
    logging.info("\n--- 最终校准结果 ---")
    logging.info(f"计算出的最终外参矩阵:\n{np.round(final_extrinsic_transform, 4)}")

    # 保存结果到 JSON 文件
    transform_dict = {
        "extrinsic_matrix": final_extrinsic_transform.tolist(),
        "fitness": reg_result.fitness,
        "inlier_rmse": reg_result.inlier_rmse
    }
    o3d.io.write_transformation(args.output_json, transform_dict)
    logging.info(f"最终外参矩阵已保存到 {args.output_json}")

    # 可视化
    if args.visualize:
        logging.info("正在可视化粗配准结果...")
        lc.draw_registration_result(
            source_cloud, target_cloud, coarse_transform, "粗配准结果"
        )
        logging.info("正在可视化最终配准结果...")
        lc.draw_registration_result(
            source_cloud, target_cloud, final_extrinsic_transform, "最终精配准结果"
        )

if __name__ == "__main__":
    main()