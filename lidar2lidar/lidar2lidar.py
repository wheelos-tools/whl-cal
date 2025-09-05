import copy
import logging

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def matrix_to_quaternion_and_translation(matrix: np.ndarray):
    """Convert a 4x4 transformation matrix to quaternion and translation.

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        tuple[list[float], np.ndarray]:
            - Quaternion in (w, x, y, z) format.
            - Translation vector (x, y, z).

    Raises:
        AssertionError: If matrix shape is not (4, 4).
    """
    assert matrix.shape == (4, 4), "Matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # (x, y, z, w)
    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
    return quat_wxyz, translation


def draw_registration_result(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray,
                             window_name: str = "Registration Result",
                             registration_result=None,
                             max_icp_corr_dist: float = 0.05):
    """Visualize registration result with correspondences.

    Args:
        source (o3d.geometry.PointCloud): Source point cloud.
        target (o3d.geometry.PointCloud): Target point cloud.
        transformation (np.ndarray): Transformation matrix.
        window_name (str): Visualization window name.
        registration_result (open3d.pipelines.registration.RegistrationResult,
            optional): Registration result with correspondences.
        max_icp_corr_dist (float): Max ICP correspondence distance.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 1, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([0, 1, 0])

    geometries = [source_temp, target_temp]
    line_points, line_indices = [], []

    if (registration_result is not None and
            hasattr(registration_result, 'correspondence_set') and
            len(registration_result.correspondence_set) > 0):
        corr = np.asarray(registration_result.correspondence_set)
        matched_source_points = np.asarray(source.points)[corr[:, 0]]
        matched_target_points = np.asarray(target.points)[corr[:, 1]]

        matched_source_points_h = np.hstack(
            (matched_source_points, np.ones((matched_source_points.shape[0], 1))))
        matched_source_points_trans = (transformation @ matched_source_points_h.T).T[:, :3]

        line_points = np.vstack((matched_source_points_trans, matched_target_points))
        line_indices = [[i, i + len(matched_source_points_trans)]
                        for i in range(len(matched_source_points_trans))]

    elif registration_result is not None:
        target_points = np.asarray(target_temp.points)
        source_points = np.asarray(source_temp.points)
        tree = o3d.geometry.KDTreeFlann(target_temp)

        for p in source_points:
            [k, idx, _] = tree.search_knn_vector_3d(p, 1)
            if k > 0 and np.linalg.norm(p - target_points[idx[0]]) < max_icp_corr_dist:
                line_points.append(p)
                line_points.append(target_points[idx[0]])
                line_indices.append([len(line_points) - 2, len(line_points) - 1])

        if len(line_indices) > 0:
            line_points = np.array(line_points)

    if len(line_indices) > 0:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in line_indices])
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float,
                           nb_neighbors: int = 20,
                           std_ratio: float = 2.0,
                           plane_dist_thresh: float = 0.05,
                           height_range: float | None = None,
                           remove_walls: bool = False,
                           wall_angle_threshold: float = 0.1,
                           max_wall_planes: int = 2) -> o3d.geometry.PointCloud:
    """Preprocess point cloud (downsample, denoise, remove ground/walls).

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling.
        nb_neighbors (int): Number of neighbors for outlier removal.
        std_ratio (float): Standard deviation ratio for outlier removal.
        plane_dist_thresh (float): Distance threshold for plane segmentation.
        height_range (float | None): Max height to keep points.
        remove_walls (bool): Whether to remove vertical walls.
        wall_angle_threshold (float): Threshold for wall detection.
        max_wall_planes (int): Maximum number of wall planes to remove.

    Returns:
        o3d.geometry.PointCloud: Preprocessed point cloud.
    """
    logging.info("  -> Cleaning invalid points...")
    points = np.asarray(pcd.points)
    mask = np.isfinite(points).all(axis=1)
    if not mask.all():
        logging.info("     Cleaning invalid points: removed %d NaN/Inf points",
                    (~mask).sum()
    )
    pcd = pcd.select_by_index(np.where(mask)[0])
    
    logging.info("  -> Input points: %d", len(pcd.points))
    
    logging.info("  -> Voxel downsampling...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    logging.info("     After downsampling: %d points", len(pcd_down.points))

    logging.info("  -> Removing statistical outliers...")
    _, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                 std_ratio=std_ratio)
    pcd_filtered = pcd_down.select_by_index(ind)
    logging.info("     After outlier removal: %d points", len(pcd_filtered.points))

    logging.info("  -> Removing ground points...")
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=plane_dist_thresh,
                                                      ransac_n=3,
                                                      num_iterations=1000)
    pcd_no_ground = pcd_filtered.select_by_index(inliers, invert=True)
    logging.info("     After ground removal: %d points", len(pcd_no_ground.points))

    if height_range is not None:
        points = np.asarray(pcd_no_ground.points)
        indices = np.where(points[:, 2] <= height_range)[0]
        pcd_no_ground = pcd_no_ground.select_by_index(indices)

    if remove_walls:
        logging.info("  -> Removing walls (vertical planes)...")
        pcd_tmp = pcd_no_ground
        for i in range(max_wall_planes):
            plane_model, inliers = pcd_tmp.segment_plane(distance_threshold=plane_dist_thresh,
                                                         ransac_n=3,
                                                         num_iterations=1000)
            a, b, c, _ = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

            if abs(normal[2]) < wall_angle_threshold:
                logging.info("    - Removed wall plane %d, points: %d", i + 1, len(inliers))
                pcd_tmp = pcd_tmp.select_by_index(inliers, invert=True)
            else:
                break
        pcd_no_ground = pcd_tmp

    logging.info("  -> Original points: %d, after preprocessing: %d", len(pcd.points), len(pcd_no_ground.points))
    logging.info("  -> Estimating normals...")
    pcd_no_ground.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_no_ground


def compute_fpfh_features(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Compute FPFH features.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size used in preprocessing.

    Returns:
        o3d.pipelines.registration.Feature: FPFH features.
    """
    logging.info("  -> Computing FPFH features...")
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))


def perform_coarse_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size: float):
    """Perform coarse registration using FPFH + RANSAC.
       Parameters should be tuned according to the specific case.

    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        source_fpfh (Feature): Source FPFH features.
        target_fpfh (Feature): Target FPFH features.
        voxel_size (float): Voxel size.

    Returns:
        o3d.pipelines.registration.RegistrationResult: Coarse registration result.
    """
    logging.info("  -> Performing coarse registration (FPFH + RANSAC)...")
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 10,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 10)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))


def perform_icp_registration(source_pcd,
                             target_pcd,
                             initial_transform: np.ndarray,
                             icp_params: dict,
                             method: int):
    """Perform iterative ICP fine registration.
       Parameters should be tuned according to the specific case.
    
    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        initial_transform (np.ndarray): Initial transformation matrix.
        icp_params (dict): ICP parameters.

    Returns:
        o3d.pipelines.registration.RegistrationResult: Final ICP registration result.
    """
    current_transform = initial_transform
    final_result = None
    for i, max_corr_dist in enumerate(icp_params['max_correspondence_distances']):
        logging.info("  -> Iteration %d: Max correspondence distance %.3f m", i + 1, max_corr_dist)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=icp_params['relative_fitness'],
            relative_rmse=icp_params['relative_rmse'],
            max_iteration=icp_params['max_iterations'][i])
        if method == 1:
            logging.info("  -> Using point-to-plane ICP")
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_corr_dist, current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria)
        elif method == 2:
            logging.info("  -> Using GICP")
            result = o3d.pipelines.registration.registration_generalized_icp(
                 source_pcd, target_pcd, max_corr_dist, current_transform,
                 o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                 criteria)
        elif method == 3:
            logging.info("  -> Using point-to-point ICP")
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_corr_dist, current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria)
        current_transform = result.transformation
        final_result = result
    return final_result


def calibrate_lidar_extrinsic(source_cloud: o3d.geometry.PointCloud,
                              target_cloud: o3d.geometry.PointCloud,
                              is_draw_registration: bool = False,
                              preprocessing_params: dict = None,
                              method: int = 1):
    """Calibrate lidar extrinsic parameters using point cloud registration.

    Args:
        source_cloud (o3d.geometry.PointCloud): Source point cloud.
        target_cloud (o3d.geometry.PointCloud): Target point cloud.
        is_draw_registration (bool): Whether to visualize results.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, o3d.pipelines.registration.RegistrationResult | None]:
        - Final extrinsic transformation matrix.
        - Initial coarse transformation matrix.
        - Final registration result.
    """
    if preprocessing_params == None:
        preprocessing_params = {
            'voxel_size': 0.04,
            'nb_neighbors': 20,
            'std_ratio': 2.0,
            'plane_dist_thresh': 0.05,
            'height_range': None,
            'remove_walls': True
        }
    icp_params = {
        'max_correspondence_distances': [1.0, 0.5, 0.1, 0.05, 0.02],
        'max_iterations': [50, 100, 100, 200, 200],
        'relative_fitness': 1e-7,
        'relative_rmse': 1e-7
    }

    logging.info("--- Step 1: Point Cloud Preprocessing ---")
    source_preprocessed = preprocess_point_cloud(source_cloud, **preprocessing_params)
    target_preprocessed = preprocess_point_cloud(target_cloud, **preprocessing_params)

    if len(source_preprocessed.points) == 0 or len(target_preprocessed.points) == 0:
        logging.error("Preprocessed point cloud is empty, calibration failed.")
        return None, None, None

    logging.info("--- Step 2: FPFH Feature Extraction ---")
    source_fpfh = compute_fpfh_features(source_preprocessed, preprocessing_params['voxel_size'])
    target_fpfh = compute_fpfh_features(target_preprocessed, preprocessing_params['voxel_size'])

    logging.info("--- Step 3: Coarse Registration (FPFH + RANSAC) ---")
    initial_guess = perform_coarse_registration(
        source_preprocessed, target_preprocessed, source_fpfh, target_fpfh, preprocessing_params['voxel_size'])
    initial_guess_transform = initial_guess.transformation

    logging.info("\n--- Coarse Calibration Metrics ---")
    logging.info("Coarse registration matched points: %d", len(initial_guess.correspondence_set))
    logging.info("  - Fitness (overlap): %.4f", initial_guess.fitness)
    logging.info("  - Inlier RMSE: %.4f", initial_guess.inlier_rmse)
    logging.info("  - Transformation Matrix:\n%s", initial_guess_transform)

    quat, trans = matrix_to_quaternion_and_translation(initial_guess_transform)
    logging.info("  - Rotation (quaternion wxyz): %s", quat)
    logging.info("  - Translation (xyz): %s", trans)

    if not np.isfinite(initial_guess_transform).all():
        logging.error("Coarse registration failed, invalid initial transformation matrix.")
        return None, None, None

    logging.info("--- Step 4: Fine Registration (Iterative GICP) ---")
    registration_result = perform_icp_registration(
        source_preprocessed, target_preprocessed, initial_guess_transform, icp_params, method)
    final_extrinsic_transform = registration_result.transformation

    if not np.isfinite(final_extrinsic_transform).all():
        logging.error("Fine registration failed, invalid transformation matrix.")
        return None, initial_guess_transform, registration_result

    logging.info("\n--- Final Calibration Metrics ---")
    source_points = np.asarray(source_preprocessed.points)
    target_points = np.asarray(target_preprocessed.points)
    tree = o3d.geometry.KDTreeFlann(target_preprocessed)
    max_icp_corr_dist = 0.4
    matched_count = sum(
        1 for p in source_points
        if tree.search_knn_vector_3d(p, 1)[0] > 0 and
        np.linalg.norm(p - target_points[tree.search_knn_vector_3d(p, 1)[1][0]]) < max_icp_corr_dist)
    logging.info("Fine registration matched points (within %.2f m): %d", max_icp_corr_dist, matched_count)
    logging.info("  - Fitness (overlap): %.4f", registration_result.fitness)
    logging.info("  - Inlier RMSE: %.4f", registration_result.inlier_rmse)
    logging.info("  - Transformation Matrix:\n%s", final_extrinsic_transform)

    quat_final, trans_final = matrix_to_quaternion_and_translation(final_extrinsic_transform)
    logging.info("  - Rotation (quaternion wxyz): %s", quat_final)
    logging.info("  - Translation (xyz): %s", trans_final)

    if is_draw_registration:
        logging.info("Visualizing coarse registration...")
        draw_registration_result(source_preprocessed, target_preprocessed,
                                 initial_guess_transform, window_name="Coarse Registration",
                                 registration_result=initial_guess)
        logging.info("Visualizing fine registration...")
        draw_registration_result(source_preprocessed, target_preprocessed,
                                 final_extrinsic_transform, window_name="Fine Registration",
                                 registration_result=registration_result,
                                 max_icp_corr_dist=max_icp_corr_dist)

    return final_extrinsic_transform, initial_guess_transform, registration_result
