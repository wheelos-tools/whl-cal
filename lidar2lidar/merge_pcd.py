import argparse
import logging

try:
    from lidar2lidar.cli import (
        load_initial_transform,
        load_point_clouds,
        save_registered_point_cloud,
    )
except ImportError:
    from cli import load_initial_transform, load_point_clouds, save_registered_point_cloud


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Merge two point clouds with a provided transform."""
    parser = argparse.ArgumentParser(
        description="Merge a source PCD into a target PCD using a JSON or YAML transform.",
    )
    parser.add_argument(
        "--source-pcd",
        type=str,
        required=True,
        help="Path to the source point cloud file.",
    )
    parser.add_argument(
        "--target-pcd",
        type=str,
        required=True,
        help="Path to the target point cloud file.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        required=True,
        help="Path to a JSON or YAML transform from source to target.",
    )
    parser.add_argument(
        "--output-pcd",
        type=str,
        default="merged_output.pcd",
        help="Path to save the merged point cloud.",
    )
    args = parser.parse_args()

    source_cloud, target_cloud = load_point_clouds(args.source_pcd, args.target_pcd)
    transform = load_initial_transform(args.transform)

    logging.info("Merging source point cloud into target frame using %s", args.transform)
    save_registered_point_cloud(source_cloud, target_cloud, transform, args.output_pcd)


if __name__ == "__main__":
    main()
