import argparse
import copy
import json
import logging
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import yaml

try:
    from lidar2lidar import lidar2lidar as lc
    from lidar2lidar.extrinsic_io import build_extrinsics_payload, load_extrinsics_file
except ImportError:
    import lidar2lidar as lc
    from extrinsic_io import build_extrinsics_payload, load_extrinsics_file

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


def detect_transform_format(path: str) -> str:
    """Detect transform serialization format from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"

    logging.error("Unsupported transform file format for %s. Use .json, .yaml, or .yml.", path)
    sys.exit(1)


def load_initial_transform(path: str) -> np.ndarray:
    """Load a 4x4 transform matrix from a JSON or YAML file.

    YAML inputs may use either the legacy compact transform form or the
    standardized extrinsics schema under `lidar2lidar/conf/`.
    """
    try:
        transform, _, _, _, _ = load_extrinsics_file(path)
    except Exception as exc:
        logging.error("Failed to load initial transform from %s: %s", path, exc)
        sys.exit(1)

    if transform.shape != (4, 4):
        logging.error("Initial transform must be a 4x4 matrix, got %s", transform.shape)
        sys.exit(1)
    if not np.isfinite(transform).all():
        logging.error("Initial transform contains non-finite values.")
        sys.exit(1)
    return transform


def build_yaml_result_payload(transformation: np.ndarray,
                              fitness: float,
                              inlier_rmse: float,
                              source_frame: str,
                              target_frame: str) -> dict:
    """Build a YAML payload using the standardized extrinsics schema."""
    return build_extrinsics_payload(
        parent_frame=target_frame,
        child_frame=source_frame,
        matrix=transformation,
        metrics={
            "fitness": float(fitness),
            "inlier_rmse": float(inlier_rmse),
        },
    )


def save_transform_result(path: str,
                          transformation: np.ndarray,
                          fitness: float,
                          inlier_rmse: float,
                          source_frame: str,
                          target_frame: str) -> None:
    """Save calibration results to JSON or YAML based on file extension."""
    transform_format = detect_transform_format(path)

    if transform_format == "json":
        payload = {
            "extrinsic_matrix": transformation.tolist(),
            "fitness": float(fitness),
            "inlier_rmse": float(inlier_rmse),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        return

    payload = build_yaml_result_payload(
        transformation,
        fitness,
        inlier_rmse,
        source_frame=source_frame,
        target_frame=target_frame,
    )
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def save_registered_point_cloud(source_cloud: o3d.geometry.PointCloud,
                                target_cloud: o3d.geometry.PointCloud,
                                transformation: np.ndarray,
                                output_path: str) -> None:
    """Save merged point cloud after applying the final transform to source."""
    transformed_source = copy.deepcopy(source_cloud)
    transformed_source.transform(transformation)
    merged_cloud = transformed_source + target_cloud
    if not o3d.io.write_point_cloud(output_path, merged_cloud):
        logging.error("Failed to save merged point cloud to %s", output_path)
        sys.exit(1)
    logging.info("Merged registered point cloud saved to %s", output_path)


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
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-json",
        dest="output_transform",
        type=str,
        default=None,
        help="Path to save the final extrinsic result. Supports .json, .yaml, and .yml.",
    )
    output_group.add_argument(
        "--output-yaml",
        dest="output_transform",
        type=str,
        default=None,
        help="Path to save the final extrinsic result as YAML.",
    )
    output_group.add_argument(
        "--output-transform",
        dest="output_transform",
        type=str,
        default=None,
        help="Path to save the final extrinsic result as JSON or YAML.",
    )
    parser.add_argument(
        "--output-pcd",
        type=str,
        default="registered_merged.pcd",
        help="Path to save the merged registered point cloud when visualization is disabled.",
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
    parser.add_argument(
        "--remove-ground",
        action="store_true",
        help="Remove the detected ground plane during preprocessing.",
    )
    parser.add_argument(
        "--remove-walls",
        action="store_true",
        help="Remove dominant vertical planes during preprocessing.",
    )
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--initial-transform-json",
        dest="initial_transform_path",
        type=str,
        default=None,
        help="Path to an initial transform file. Supports JSON matrices and standardized YAML extrinsics files.",
    )
    initial_group.add_argument(
        "--initial-transform-yaml",
        dest="initial_transform_path",
        type=str,
        default=None,
        help="Path to an initial transform YAML file, preferably from lidar2lidar/conf/*.yaml.",
    )
    initial_group.add_argument(
        "--initial-transform",
        dest="initial_transform_path",
        type=str,
        default=None,
        help="Path to an initial transform file. Prefer standardized YAML extrinsics files under lidar2lidar/conf/.",
    )
    args = parser.parse_args()

    if args.output_transform is None:
        args.output_transform = "calibration_result.json"

    # Load point clouds
    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)
    initial_transform = (
        load_initial_transform(args.initial_transform_path)
        if args.initial_transform_path is not None else None
    )

    # Modify the original parameters
    preprocessing_params = {
        'voxel_size': args.voxel_size,
        'nb_neighbors': 20,
        'std_ratio': 2.0,
        'plane_dist_thresh': 0.05,
        'height_range': args.max_height,
        'remove_ground': args.remove_ground,
        'remove_walls': args.remove_walls
    }

    # Load the method
    method = args.method

    # Perform calibration
    logging.info("--- Starting LiDAR extrinsic calibration ---")
    final_extrinsic_transform, coarse_transform, reg_result = lc.calibrate_lidar_extrinsic(
        source_cloud, target_cloud, args.visualize, preprocessing_params, method, initial_transform
    )

    if final_extrinsic_transform is None:
        logging.error("Calibration failed.")
        return

    # Print results
    logging.info("\n--- Final calibration result ---")
    logging.info("Computed final extrinsic matrix:\n%s", format_matrix_for_logging(final_extrinsic_transform))
    logging.info("Fitness: %.6f", reg_result.fitness)
    logging.info("Inlier RMSE: %.6f", reg_result.inlier_rmse)

    # Save result to JSON or YAML
    save_transform_result(
        args.output_transform,
        final_extrinsic_transform,
        reg_result.fitness,
        reg_result.inlier_rmse,
        source_frame=Path(args.source_pcd).stem,
        target_frame=Path(args.target_pcd).stem,
    )

    logging.info("Final extrinsic matrix saved to %s", args.output_transform)

    if not args.visualize:
        save_registered_point_cloud(
            source_cloud,
            target_cloud,
            final_extrinsic_transform,
            args.output_pcd,
        )


if __name__ == "__main__":
    main()
