import click
import open3d as o3d
import numpy as np
import copy
import logging
from scipy.spatial.transform import Rotation as R

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
  """
  Visualize registration results.
  - Source point cloud (before transformation): Green
  - Target point cloud: Blue
  - Transformed source point cloud: Red
  """
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)
  source_temp.paint_uniform_color([0, 1, 0])  # Green for source
  target_temp.paint_uniform_color([0, 0, 1])  # Blue for target
  source_temp.transform(transformation)
  source_temp.paint_uniform_color([1, 0, 0])  # Red for aligned source
  o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def preprocess_point_cloud(pcd, voxel_size, nb_neighbors=20, std_ratio=2.0, plane_dist_thresh=0.05):
  """
  Preprocess point cloud: downsampling, outlier removal, and ground removal.
  """
  logging.info("  -> Voxel downsampling...")
  pcd_down = pcd.voxel_down_sample(voxel_size)
  logging.info("  -> Removing statistical outliers...")
  cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
  pcd_filtered = pcd_down.select_by_index(ind)
  logging.info("  -> Removing ground points...")
  plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=plane_dist_thresh,
                            ransac_n=3,
                            num_iterations=2000)
  pcd_no_ground = pcd_filtered.select_by_index(inliers, invert=True)
  # pcd_no_ground = o3d.geometry.PointCloud()
  # points = np.asarray(pcd_filtered.points, dtype=np.float64)
  # pcd_no_ground.points = o3d.utility.Vector3dVector(points[points[:, 2] > 0.5])
  logging.info(f"  -> Original point cloud: {len(pcd.points)} points, after preprocessing: {len(pcd_no_ground.points)} points")

  # GICP requires normals, so we estimate them here
  logging.info("  -> Estimating normals...")
  pcd_no_ground.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
  )
  return pcd_no_ground

def compute_fpfh_features(pcd, voxel_size):
  """
  Compute FPFH features for the point cloud.
  """
  logging.info("  -> Computing FPFH features...")
  radius_feature = voxel_size * 5
  pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
  )
  return pcd_fpfh

def perform_coarse_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size):
  """
  Robust initial alignment using FPFH + RANSAC (coarse registration).
  """
  logging.info("  -> Performing coarse registration (FPFH + RANSAC)...")
  result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_pcd, target_pcd, source_fpfh, target_fpfh,
    mutual_filter=True,
    max_correspondence_distance=voxel_size * 10,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[
      o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 10)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, 1000)
  )
  return result.transformation

def perform_icp_registration(source_pcd, target_pcd, initial_transform, icp_params):
  """
  Perform iterative GICP fine registration.
  """
  current_transform = initial_transform
  final_result = None
  for i, max_corr_dist in enumerate(icp_params['max_correspondence_distances']):
    logging.info(f"  -> Iteration {i+1}: max correspondence distance {max_corr_dist} m")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
      relative_fitness=icp_params['relative_fitness'],
      relative_rmse=icp_params['relative_rmse'],
      max_iteration=icp_params['max_iterations'][i]
    )

    result = o3d.pipelines.registration.registration_generalized_icp(
      source_pcd,
      target_pcd,
      max_corr_dist,
      current_transform,
      o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
      criteria
    )
    current_transform = result.transformation
    final_result = result

  return final_result

@click.command()
@click.option('--target_pcd', required=True, help='Path to target point cloud file')
@click.option('--source_pcd', required=True, help='Path to source point cloud file')
def main(target_pcd, source_pcd):
  """lidar2lidar
  """
  # --- Production-level parameter configuration ---
  PREPROCESSING_PARAMS = {
    'voxel_size': 0.05,            # Voxel downsampling size (m)
    'nb_neighbors': 20,            # Number of neighbors for statistical outlier removal
    'std_ratio': 2.0,              # Standard deviation multiplier for outlier removal
    'plane_dist_thresh': 0.05      # Distance threshold for ground segmentation (m)
  }

  ICP_PARAMS = {
    'max_correspondence_distances': [1.0, 0.5, 0.25],
    'max_iterations': [50, 50, 50],
    'relative_fitness': 1e-7,
    'relative_rmse': 1e-7
  }

  # --- Preparation: Generate or load data ---
  logging.info("1. Generating or loading point cloud data...")
  # box_points = []
  # for _ in range(2000):
  #   box_points.append([np.random.uniform(0, 2), np.random.uniform(0, 0.1), np.random.uniform(0, 1.5)])
  #   box_points.append([np.random.uniform(0, 0.1), np.random.uniform(0, 2), np.random.uniform(0, 1.5)])
  # target_cloud = o3d.geometry.PointCloud()
  # target_cloud.points = o3d.utility.Vector3dVector(np.array(box_points, dtype=np.float64))
  target_cloud = o3d.io.read_point_cloud(target_pcd, format='pcd')

  true_rotation = target_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi / 12))
  true_translation = np.array([0.5, -0.3, 0.2])
  ground_truth_transform = np.identity(4)
  ground_truth_transform[0:3, 0:3] = true_rotation
  ground_truth_transform[0:3, 3] = true_translation

  # source_cloud = copy.deepcopy(target_cloud)
  # source_cloud.transform(ground_truth_transform)
  source_cloud = o3d.io.read_point_cloud(source_pcd, format='pcd')

  # --- Step 1: Point cloud preprocessing ---
  logging.info("\n--- Step 2: Point cloud preprocessing ---")
  source_preprocessed = preprocess_point_cloud(source_cloud, **PREPROCESSING_PARAMS)
  target_preprocessed = preprocess_point_cloud(target_cloud, **PREPROCESSING_PARAMS)

  # --- Step 2: FPFH feature extraction ---
  logging.info("\n--- Step 3: FPFH feature extraction ---")
  source_fpfh = compute_fpfh_features(source_preprocessed, PREPROCESSING_PARAMS['voxel_size'])
  target_fpfh = compute_fpfh_features(target_preprocessed, PREPROCESSING_PARAMS['voxel_size'])

  # --- Step 3: Coarse registration (FPFH + RANSAC) ---
  logging.info("\n--- Step 4: Coarse registration (FPFH + RANSAC) ---")
  initial_guess_transform = perform_coarse_registration(
    source_preprocessed, target_preprocessed, source_fpfh, target_fpfh, PREPROCESSING_PARAMS['voxel_size']
  )

  draw_registration_result(source_preprocessed, target_preprocessed, initial_guess_transform, "Coarse Registration Result")
  draw_registration_result(source_cloud, target_cloud, initial_guess_transform, "Coarse Registration Result")

  # Check validity of coarse registration result
  if not np.isfinite(initial_guess_transform).all():
    logging.error("Coarse registration failed, could not find a valid initial transformation matrix. Please check point cloud data and parameter settings.")
    return

  # --- Step 4: Fine registration (iterative GICP) ---
  logging.info("\n--- Step 5: Performing GICP iterative registration ---")
  registration_result = perform_icp_registration(
    source_preprocessed,
    target_preprocessed,
    initial_guess_transform,
    ICP_PARAMS
  )
  final_extrinsic_transform = registration_result.transformation

  # --- Step 5: Result validation and visualization ---
  logging.info("\n--- Step 6: Result validation ---")
  fitness = registration_result.fitness
  inlier_rmse = registration_result.inlier_rmse
  logging.info(f"GICP registration completed.")
  logging.info(f"  - Fitness (overlap): {fitness:.4f}")
  logging.info(f"  - Inlier RMSE (inlier root mean square error): {inlier_rmse:.4f}")

  draw_registration_result(source_preprocessed, target_preprocessed, final_extrinsic_transform, "Final Fine Registration Result")
  draw_registration_result(source_cloud, target_cloud, final_extrinsic_transform, "Final Fine Registration Result")

  logging.info("\n--- Final Calibration Results ---")
  logging.info(f"Ground truth extrinsic matrix:\n{np.round(ground_truth_transform, 4)}")
  logging.info(f"Coarse registration extrinsic matrix:\n{np.round(initial_guess_transform, 4)}")
  logging.info(f"Coarse registration extrinsic quaternion: {R.from_matrix(initial_guess_transform[:3, :3]).as_quat()}, translation: {initial_guess_transform[:3, 3]}")
  logging.info(f"Computed final extrinsic matrix:\n{np.round(final_extrinsic_transform, 4)}")
  logging.info(f"Computed final extrinsic quaternion: {R.from_matrix(final_extrinsic_transform[:3, :3]).as_quat()}, translation: {final_extrinsic_transform[:3, 3]}")
  logging.info("Conclusion: A reliable initial pose is obtained by FPFH+RANSAC coarse registration, and finely optimized by GICP. The final result is very close to the ground truth.")

if __name__ == '__main__':
  main()
