import open3d as o3d
import numpy as np
import logging
import copy

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
    """
    Visualize registration results.
    - Source point cloud (before transformation): Green
    - Target point cloud: Blue
    - Transformed source point cloud: Red
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def preprocess_point_cloud(pcd, voxel_size, nb_neighbors=20, std_ratio=2.0, plane_dist_thresh=0.05):
    """
    Preprocess the point cloud: downsampling, outlier removal, and ground removal.
    """
    logging.info("  -> Voxel downsampling...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    logging.info("  -> Removing statistical outliers...")
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_filtered = pcd_down.select_by_index(ind)
    logging.info("  -> Removing ground points...")
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=plane_dist_thresh,
                                                      ransac_n=3,
                                                      num_iterations=1000)
    pcd_no_ground = pcd_filtered.select_by_index(inliers, invert=True)
    logging.info(f"  -> Original point cloud: {len(pcd.points)} points, after preprocessing: {len(pcd_no_ground.points)} points")
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
    Robust initial alignment (coarse registration) using FPFH + RANSAC.
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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result.transformation

def perform_icp_registration(source_pcd, target_pcd, initial_transform, icp_params):
    """
    Perform iterative GICP fine registration.
    """
    current_transform = initial_transform
    final_result = None
    for i, max_corr_dist in enumerate(icp_params['max_correspondence_distances']):
        logging.info(f"  -> Iteration {i+1}: Max correspondence distance {max_corr_dist} m")
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

def calibrate_lidar_extrinsic(source_cloud, target_cloud):
    """
    Main function for LiDAR-to-LiDAR extrinsic calibration.
    Encapsulates the complete calibration process, from preprocessing to final fine registration.

    Args:
        source_cloud (o3d.geometry.PointCloud): Point cloud from source LiDAR.
        target_cloud (o3d.geometry.PointCloud): Point cloud from target LiDAR.

    Returns:
        tuple: (final_extrinsic_transform, initial_guess_transform, registration_result)
        final_extrinsic_transform (np.ndarray): Final 4x4 extrinsic matrix.
        initial_guess_transform (np.ndarray): Initial 4x4 transformation matrix from coarse registration.
        registration_result (o3d.pipelines.registration.RegistrationResult): Result object containing fine registration metrics (fitness, inlier_rmse).
    """
    PREPROCESSING_PARAMS = {
        'voxel_size': 0.02,
        'nb_neighbors': 20,
        'std_ratio': 2.0,
        'plane_dist_thresh': 0.05
    }
    ICP_PARAMS = {
        'max_correspondence_distances': [1.0, 0.5, 0.1, 0.05, 0.02],
        'max_iterations': [50, 100, 100, 200, 200],
        'relative_fitness': 1e-7,
        'relative_rmse': 1e-7
    }

    logging.info("--- Step 1: Point Cloud Preprocessing ---")
    source_preprocessed = preprocess_point_cloud(source_cloud, **PREPROCESSING_PARAMS)
    target_preprocessed = preprocess_point_cloud(target_cloud, **PREPROCESSING_PARAMS)

    if len(source_preprocessed.points) == 0 or len(target_preprocessed.points) == 0:
        logging.error("Preprocessed point cloud is empty, calibration failed.")
        return None, None, None

    logging.info("--- Step 2: FPFH Feature Extraction ---")
    source_fpfh = compute_fpfh_features(source_preprocessed, PREPROCESSING_PARAMS['voxel_size'])
    target_fpfh = compute_fpfh_features(target_preprocessed, PREPROCESSING_PARAMS['voxel_size'])

    logging.info("--- Step 3: Coarse Registration (FPFH + RANSAC) ---")
    initial_guess_transform = perform_coarse_registration(
        source_preprocessed, target_preprocessed, source_fpfh, target_fpfh, PREPROCESSING_PARAMS['voxel_size']
    )

    if not np.isfinite(initial_guess_transform).all():
        logging.error("Coarse registration failed, could not find a valid initial transformation matrix. Returning None.")
        return None, None, None

    logging.info("--- Step 4: Fine Registration (Iterative GICP) ---")
    registration_result = perform_icp_registration(
        source_preprocessed,
        target_preprocessed,
        initial_guess_transform,
        ICP_PARAMS
    )
    final_extrinsic_transform = registration_result.transformation

    if not np.isfinite(final_extrinsic_transform).all():
        logging.error("Fine registration failed, could not obtain a valid transformation matrix. Returning None.")
        return None, initial_guess_transform, registration_result

    # Additional evaluation metrics output
    logging.info("\n--- Final Calibration Metrics ---")
    logging.info(f"  - Fitness (overlap): {registration_result.fitness:.4f}")
    logging.info(f"  - Inlier RMSE: {registration_result.inlier_rmse:.4f}")

    return final_extrinsic_transform, initial_guess_transform, registration_result
