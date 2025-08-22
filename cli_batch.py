#!/usr/bin/env python3
import os
import logging
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R
from lidar2lidar import calibrate_lidar_extrinsic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Converts a 4x4 rotation matrix to a quaternion [x, y, z, w]."""
    return R.from_matrix(mat[:3, :3]).as_quat()


def quaternion_to_matrix(quat: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Converts a quaternion and translation vector to a 4x4 transformation matrix."""
    rot = R.from_quat(quat).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    return mat


def matrix_to_ros_yaml(mat: np.ndarray) -> dict:
    """Converts a 4x4 matrix to ROS YAML style dictionary."""
    t = mat[:3, 3]
    quat = matrix_to_quaternion(mat)
    return {
        "transform": {
            "translation": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "rotation": {"x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]), "w": float(quat[3])}
        }
    }


def save_yaml_matrix(matrix: np.ndarray, yaml_path: str):
    """Saves a single 4x4 transformation matrix as a ROS YAML file."""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(matrix_to_ros_yaml(matrix), f, default_flow_style=False, sort_keys=False)
    logging.info("Saved YAML: %s", yaml_path)


def average_transform_matrices_weighted(matrices: list[np.ndarray], fitnesses: list[float]) -> np.ndarray:
    """Computes weighted average of 4x4 transformation matrices using quaternion averaging."""
    quats = np.array([matrix_to_quaternion(m) for m in matrices])
    translations = np.array([m[:3, 3] for m in matrices])
    weights = np.array(fitnesses) / np.sum(fitnesses)

    # Weighted quaternion averaging
    A = np.zeros((4, 4))
    for q, w in zip(quats, weights):
        q /= np.linalg.norm(q)
        A += w * np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    avg_quat = eigvecs[:, np.argmax(eigvals)]

    # Weighted translation averaging
    avg_translation = np.average(translations, axis=0, weights=weights)
    return quaternion_to_matrix(avg_quat, avg_translation)


def filter_and_average_results(matrices: list[np.ndarray], metrics_list: list[dict]) -> np.ndarray | None:
    """Filters bad frames and computes weighted average of valid matrices."""
    df = pd.DataFrame(metrics_list)
    fitness_mean, fitness_std = df["fine_fitness"].mean(), df["fine_fitness"].std()
    rmse_mean, rmse_std = df["fine_rmse"].mean(), df["fine_rmse"].std()
    fitness_thresh = fitness_mean - fitness_std
    rmse_thresh = rmse_mean + rmse_std

    logging.info("Adaptive thresholds: fitness >= %.4f, rmse <= %.4f", fitness_thresh, rmse_thresh)

    valid_mats, valid_fitness = [], []
    for m, row in zip(matrices, metrics_list):
        if not np.isfinite(m).all():
            logging.warning("Frame %d discarded: invalid matrix", row['frame'])
            continue
        if row["fine_fitness"] >= fitness_thresh and row["fine_rmse"] <= rmse_thresh:
            valid_mats.append(m)
            valid_fitness.append(row["fine_fitness"])
        else:
            logging.warning(
                "Frame %d discarded: fitness=%.3f, rmse=%.3f",
                row['frame'], row['fine_fitness'], row['fine_rmse']
            )

    if not valid_mats:
        logging.error("All frames discarded. Calibration failed.")
        return None

    return average_transform_matrices_weighted(valid_mats, valid_fitness)


def load_pointcloud_files(folder: str) -> list[str]:
    """Loads all PCD files from a folder."""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".pcd")]


def calibrate_pair(source_folder: str, target_folder: str, visualize: bool = False, max_frames: int = 10) -> np.ndarray | None:
    """Calibrates a single LiDAR pair and returns averaged transformation matrix with progress logging."""
    source_files = load_pointcloud_files(source_folder)
    target_files = load_pointcloud_files(target_folder)
    num_frames = min(len(source_files), len(target_files), max_frames)

    if num_frames == 0:
        logging.warning("No point cloud files found in %s or %s", source_folder, target_folder)
        return None

    matrices, metrics_list = [], []

    logging.info("Total frames to process for this pair: %d", num_frames)
    for i in range(num_frames):
        progress = (i + 1) / num_frames * 100
        logging.info("Processing frame %d/%d (%.1f%%) for pair %s → %s",
                     i + 1, num_frames, progress, os.path.basename(source_folder), os.path.basename(target_folder))
        src_pcd = o3d.io.read_point_cloud(source_files[i])
        tgt_pcd = o3d.io.read_point_cloud(target_files[i])
        try:
            final_tf, initial_guess, reg_result = calibrate_lidar_extrinsic(
                src_pcd, tgt_pcd, is_draw_registration=visualize
            )
        except Exception as e:
            logging.error("Frame %d exception: %s", i + 1, e)
            continue
        if final_tf is None:
            logging.warning("Frame %d calibration failed. Skipping.", i + 1)
            continue

        matrices.append(final_tf)
        coarse_fitness = getattr(initial_guess, "fitness", 0.0)
        fine_fitness = getattr(reg_result, "fitness", 0.0)
        fine_rmse = getattr(reg_result, "inlier_rmse", np.nan)

        metrics_list.append({
            "frame": i + 1,
            "coarse_fitness": coarse_fitness,
            "coarse_rmse": getattr(initial_guess, "inlier_rmse", np.nan),
            "fine_fitness": fine_fitness,
            "fine_rmse": fine_rmse
        })
        logging.info("Frame %d done: fine_fitness=%.4f, fine_rmse=%.4f", i + 1, fine_fitness, fine_rmse)

    if not matrices:
        logging.error("No valid frames for pair: %s → %s", source_folder, target_folder)
        return None

    logging.info("Filtering and averaging results for pair %s → %s", source_folder, target_folder)
    return filter_and_average_results(matrices, metrics_list)


def main():
    """Entry point for batch calibration of four LiDAR pairs using a parent folder."""
    parser = argparse.ArgumentParser(description="Batch LiDAR calibration for four fixed pairs")
    parser.add_argument(
        "--parent_dir",
        required=True,
        help="Parent folder containing left_back, right_back, right_front, left_front subfolders"
    )
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max_frames", type=int, default=10)
    args = parser.parse_args()

    lidar_names = ["left_back", "right_back", "right_front", "left_front"]
    pairs = [
        ("left_back", "right_back"),
        ("right_back", "right_front"),
        ("right_front", "left_front"),
        ("left_front", "left_back")
    ]

    # Verify all required folders exist
    folder_map = {}
    for name in lidar_names:
        path = os.path.join(args.parent_dir, name)
        if not os.path.isdir(path):
            logging.error("Required LiDAR folder does not exist: %s", path)
            return
        folder_map[name] = path
        logging.info("Found LiDAR folder: %s", path)

    os.makedirs(args.output_dir, exist_ok=True)
    final_txt_path = os.path.join(args.output_dir, "final.txt")

    with open(final_txt_path, "w") as f:
        for idx, (src_name, tgt_name) in enumerate(pairs, start=1):
            logging.info("=== Processing pair %d/%d: %s → %s ===", idx, len(pairs), src_name, tgt_name)
            src_folder = folder_map[src_name]
            tgt_folder = folder_map[tgt_name]
            matrix = calibrate_pair(src_folder, tgt_folder, visualize=args.visualize, max_frames=args.max_frames)
            if matrix is None:
                logging.error("Calibration failed for pair %s → %s", src_name, tgt_name)
                continue
            yaml_path = os.path.join(args.output_dir, f"{src_name}2{tgt_name}.yaml")
            save_yaml_matrix(matrix, yaml_path)
            f.write(f"{src_name}2{tgt_name} = [\n")
            for row in matrix:
                f.write(f"    {row.tolist()},\n")
            f.write("]\n\n")
            logging.info("Saved matrix for %s → %s", src_name, tgt_name)

    logging.info("All pairs processed. final.txt saved to %s", final_txt_path)


if __name__ == "__main__":
    main()
