import argparse
import logging
import sys
import json
import numpy as np
import open3d as o3d
import lidar2lidar as lc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_point_clouds(source_path: str, target_path: str) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Load point clouds from specified file paths.

    Args:
        source_path (str): Path to the source point cloud file.
        target_path (str): Path to the target point cloud file.

    Returns:
        tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
            Loaded source and target point clouds.

    Raises:
        SystemExit: If either point cloud cannot be loaded or is empty.
    """
    logging.info("Loading point cloud files: %s and %s", source_path, target_path)
    source_cloud = o3d.io.read_point_cloud(source_path)
    target_cloud = o3d.io.read_point_cloud(target_path)

    if not source_cloud.has_points() or not target_cloud.has_points():
        logging.error("Failed to load point cloud files or point clouds are empty.")
        sys.exit(1)

    return source_cloud, target_cloud


def format_matrix_for_logging(matrix: np.ndarray) -> str:
    """Format a 4x4 transformation matrix as a NumPy-style string.

    Args:
        matrix (np.ndarray): Transformation matrix.

    Returns:
        str: Formatted matrix string.
    """
    return np.array2string(matrix, precision=6, suppress_small=True)


def main() -> None:
    """Command-line tool for LiDAR-to-LiDAR extrinsic calibration."""
    parser = argparse.ArgumentParser(description="LiDAR to LiDAR Extrinsic Calibration Tool.")
    parser.add_argument(
        "--source-pcd",
        type=str,
        required=True,
        help="Path to the source LiDAR point cloud file (e.g., lidar1.pcd).",
    )
    parser.add_argument(
        "--target-pcd",
        type=str,
        required=True,
        help="Path to the target LiDAR point cloud file (e.g., lidar2.pcd).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="calibration_result.json",
        help="Path to save the final extrinsic matrix.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the registration results.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="Voxel size for point cloud downsampling.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Maximum height to keep points (height_range in preprocessing).",
    )
    parser.add_argument(
        "--method",
        type=int,
        default=1,
        help="Method for registration (1: point_to_plane, 2: GICP, 3: point_to_point).",
    )
    args = parser.parse_args()

    # Load point clouds
    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)

    # Modify the original parameters
    preprocessing_params = {
        'voxel_size': args.voxel_size,
        'nb_neighbors': 20,
        'std_ratio': 2.0,
        'plane_dist_thresh': 0.05,
        'height_range': args.max_height,
        'remove_walls': True
    }

    # Load the method
    method = args.method

    # Perform calibration
    logging.info("--- Starting LiDAR extrinsic calibration ---")
    final_extrinsic_transform, coarse_transform, reg_result = lc.calibrate_lidar_extrinsic(
        source_cloud, target_cloud, args.visualize, preprocessing_params, method
    )

    if final_extrinsic_transform is None:
        logging.error("Calibration failed.")
        return

    # Print results
    logging.info("\n--- Final calibration result ---")
    logging.info("Computed final extrinsic matrix:\n%s", format_matrix_for_logging(final_extrinsic_transform))
    logging.info("Fitness: %.6f", reg_result.fitness)
    logging.info("Inlier RMSE: %.6f", reg_result.inlier_rmse)

    # Save result to JSON
    transform_dict = {
        "extrinsic_matrix": final_extrinsic_transform.tolist(),
        "fitness": reg_result.fitness,
        "inlier_rmse": reg_result.inlier_rmse,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(transform_dict, f, indent=4)

    logging.info("Final extrinsic matrix saved to %s", args.output_json)



if __name__ == "__main__":
    main()
