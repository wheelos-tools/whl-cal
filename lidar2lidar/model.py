import open3d as o3d
import numpy as np
import os
import argparse
import logging

from lidar2lidar import calibrate_lidar_extrinsic, draw_registration_result

def build_model_from_pcds(pcd_folder: str,  preprocessing_params,
                           output_filename: str = "final_vehicle_model.pcd",
                           visualize_steps: bool = True):
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
        final_transform, _, _ = calibrate_lidar_extrinsic(
            source_cloud=source_cloud,
            target_cloud=target_cloud,
            is_draw_registration=False,
            preprocessing_params=preprocessing_params,
            method=1 # using point-to-plane ICP
        )

        if final_transform is None:
            logging.warning(f"frame {source_file} mapping failed, skip")
            continue
        
        logging.info(f"frame {source_file} succeed")
        
        # transform
        source_cloud.transform(final_transform)
        accumulated_pcd += source_cloud

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
    build_model_from_pcds(pcd_folder_path, preprocessing_params, visualize_steps=False)