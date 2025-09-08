import open3d as o3d
import numpy as np
import os
import argparse
import logging

from lidar2lidar import calibrate_lidar_extrinsic, draw_registration_result

def build_model_from_pcds(pcd_folder: str,  preprocessing_params,
                           output_filename: str = "final_vehicle_model.pcd",
                           visualize_steps: bool = False,
                           skip_coarse_after_first: bool = True,
                           coarse_method: str = "auto",
                           coarse_voxel_factor: float = 2.5,
                           min_fitness: float = 0.10,
                           max_rmse: float = 0.05,
                           merge: bool = True,
                           max_frames: int | None = None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith('.pcd')])
        if not pcd_files:
            logging.error(f"no pcd on '{pcd_folder}'")
            return
    except FileNotFoundError:
        logging.error(f"'{pcd_folder}' not existed")
        return

    logging.info(f"find {len(pcd_files)} pcd files")

    try:
        accumulated_pcd = o3d.io.read_point_cloud(pcd_files[0])
        logging.info(f"init: {pcd_files[0]}, points number: {len(accumulated_pcd.points)}")
    except Exception as e:
        logging.error(f"load {pcd_files[0]} failed: {e}")
        return

    last_transform = np.eye(4)

    if max_frames is not None:
        pcd_files = pcd_files[:max(1, int(max_frames))]

    for i in range(1, len(pcd_files)):
        source_file = pcd_files[i]
        logging.info(f"\n{'='*20} processing {i+1}/{len(pcd_files)} frame: {source_file} {'='*20}")

        try:
            source_cloud = o3d.io.read_point_cloud(source_file)
            if not source_cloud.has_points():
                logging.warning(f"file {source_file} is empty, skip")
                continue
        except Exception as e:
            logging.warning(f"load {source_file} failed: {e}, skip")
            continue

        target_cloud = accumulated_pcd

        logging.info("new mapping...")
        # Use previous transform as initial guess to skip slow coarse matching
        initial_guess = last_transform if skip_coarse_after_first and i > 1 else None

        final_transform, _, reg_result = calibrate_lidar_extrinsic(
            source_cloud=source_cloud,
            target_cloud=target_cloud,
            is_draw_registration=False,
            preprocessing_params=preprocessing_params,
            method=1, # using point-to-plane ICP
            initial_transform=initial_guess,
            coarse_method=coarse_method,
            coarse_voxel_factor=coarse_voxel_factor,
            quality_gate={'min_fitness': min_fitness, 'max_rmse': max_rmse, 'max_translation': 3.0, 'max_rotation_deg': 20.0},
            target_light_preprocess=True
        )

        if final_transform is None:
            logging.warning(f"frame {source_file} mapping failed, skip")
            continue
        
        if reg_result is None:
            logging.warning(f"frame {source_file} has no registration result, skip")
            continue

        if reg_result.fitness < min_fitness or reg_result.inlier_rmse > max_rmse:
            logging.warning(f"frame {source_file} quality gate failed (fitness={reg_result.fitness:.3f}, rmse={reg_result.inlier_rmse:.3f}), skip")
            continue

        logging.info(f"frame {source_file} succeed")
        
        # Update last transform
        last_transform = final_transform

        # transform and merge
        source_cloud.transform(final_transform)
        if merge:
            accumulated_pcd += source_cloud
        else:
            accumulated_pcd = source_cloud

        # 为了防止点云过于密集导致后续配准变慢，可以在合并后进行一次下采样
        accumulated_pcd = accumulated_pcd.voxel_down_sample(preprocessing_params['voxel_size'])
        logging.info(f"total point: {len(accumulated_pcd.points)}")

        if visualize_steps:
            o3d.visualization.draw_geometries(
              [accumulated_pcd], window_name=f"accumulated_pcd - step {i+1}")

    logging.info("\ndone")
    logging.info(f"point of final model: {len(accumulated_pcd.points)}")
    
    # clear
    # logging.info("clearing...")
    # cl, ind = accumulated_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    final_model = accumulated_pcd #.select_by_index(ind)

    logging.info(f"final point after clearing: {len(final_model.points)}")

    o3d.io.write_point_cloud(output_filename, final_model)
    logging.info(f"save to: {output_filename}")

    # visualization
    # optional visualization
    # if visualize_steps:
    logging.info("show model...")
    o3d.visualization.draw_geometries([final_model], window_name="final model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model creation")
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="Path to the pcd file dir (e.g., lidar1.pcd).",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.04,
        help="Voxel size for point cloud downsampling.",
    )
    parser.add_argument(
        "--max-height", type=float, default=None,
        help="Maximum height to keep points (height_range in preprocessing).",
    )
    parser.add_argument(
        "--method", type=int, default=1,
        help="Method for registration (1: point_to_plane, 2: GICP, 3: point_to_point).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize intermediate/final model windows.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit the number of frames used for model building (for quick tests).",
    )
    parser.add_argument(
        "--skip-coarse-after-first", action="store_true",
        help="Skip coarse matching after first frame by using previous transform as initial guess.",
    )
    parser.add_argument(
        "--coarse-method", type=str, default="auto", choices=["auto", "fast", "ransac"],
        help="Coarse matching method: auto/fast/ransac.",
    )
    parser.add_argument(
        "--coarse-voxel-factor", type=float, default=2.5,
        help="Coarse-level voxel size factor relative to --voxel-size.",
    )
    parser.add_argument(
        "--min-fitness", type=float, default=0.10,
        help="Minimum fitness to accept a frame into the model.",
    )
    parser.add_argument(
        "--max-rmse", type=float, default=0.05,
        help="Maximum inlier RMSE to accept a frame into the model.",
    )
    args = parser.parse_args()

    preprocessing_params = {
        'voxel_size': args.voxel_size,
        'nb_neighbors': 20,
        'std_ratio': 2.0,
        'plane_dist_thresh': 0.05,
        'height_range': None,
        'remove_walls': True
    }

    pcd_folder_path = args.input_dir
    build_model_from_pcds(
        pcd_folder_path,
        preprocessing_params,
        visualize_steps=args.visualize,
        skip_coarse_after_first=args.skip_coarse_after_first,
        coarse_method=args.coarse_method,
        coarse_voxel_factor=args.coarse_voxel_factor,
        min_fitness=args.min_fitness,
        max_rmse=args.max_rmse,
        max_frames=args.max_frames,
    )
