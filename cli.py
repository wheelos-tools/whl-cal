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
    logging.info(f"Loading point cloud files: {source_path} and {target_path}")
    source_cloud = o3d.io.read_point_cloud(source_path)
    target_cloud = o3d.io.read_point_cloud(target_path)

    if not source_cloud.has_points() or not target_cloud.has_points():
        logging.error("Failed to load point cloud files or point clouds are empty.")
        sys.exit(1)

    return source_cloud, target_cloud

def main():
    parser = argparse.ArgumentParser(description="LiDAR to LiDAR Extrinsic Calibration Tool.")
    parser.add_argument("--source-pcd", type=str, required=True, help="Path to the source LiDAR point cloud file (e.g., lidar1.pcd).")
    parser.add_argument("--target-pcd", type=str, required=True, help="Path to the target LiDAR point cloud file (e.g., lidar2.pcd).")
    parser.add_argument("--output-json", type=str, default="calibration_result.json", help="Path to save the final extrinsic matrix.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the registration results.")

    args = parser.parse_args()

    # Load point clouds
    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)

    # Perform calibration
    logging.info("--- Starting LiDAR extrinsic calibration ---")
    final_extrinsic_transform, coarse_transform, reg_result = lc.calibrate_lidar_extrinsic(
        source_cloud, target_cloud
    )

    if final_extrinsic_transform is None:
        logging.error("Calibration failed.")
        return

    # Print and save results
    logging.info("\n--- Final calibration result ---")
    logging.info(f"Computed final extrinsic matrix:\n{np.round(final_extrinsic_transform, 4)}")

    # Save result to JSON file
    transform_dict = {
        "extrinsic_matrix": final_extrinsic_transform.tolist(),
        "fitness": reg_result.fitness,
        "inlier_rmse": reg_result.inlier_rmse
    }
    o3d.io.write_transformation(args.output_json, transform_dict)
    logging.info(f"Final extrinsic matrix saved to {args.output_json}")

    # Visualization
    if args.visualize:
        logging.info("Visualizing coarse registration result...")
        lc.draw_registration_result(
            source_cloud, target_cloud, coarse_transform, "Coarse Registration Result"
        )
        logging.info("Visualizing final registration result...")
        lc.draw_registration_result(
            source_cloud, target_cloud, final_extrinsic_transform, "Final Fine Registration Result"
        )

if __name__ == "__main__":
    main()
