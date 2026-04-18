#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-02-09
# Author: daohu527

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
        corr = np.asarray(registration_result.correspondence_set, dtype=int)
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        valid_mask = (
            (corr[:, 0] >= 0) & (corr[:, 0] < len(source_points)) &
            (corr[:, 1] >= 0) & (corr[:, 1] < len(target_points))
        )

        if not valid_mask.all():
            invalid_count = int((~valid_mask).sum())
            logging.warning(
                "Skipping %d out-of-range correspondences during visualization.",
                invalid_count,
            )
            corr = corr[valid_mask]

        if len(corr) > 0:
            matched_source_points = source_points[corr[:, 0]]
            matched_target_points = target_points[corr[:, 1]]

            matched_source_points_h = np.hstack(
                (matched_source_points, np.ones((matched_source_points.shape[0], 1))))
            matched_source_points_trans = (transformation @ matched_source_points_h.T).T[:, :3]

            line_points = np.vstack((matched_source_points_trans, matched_target_points))
            line_indices = [[i, i + len(matched_source_points_trans)]
                            for i in range(len(matched_source_points_trans))]

    if registration_result is not None and len(line_indices) == 0:
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
                           remove_ground: bool = False,
                           remove_walls: bool = False,
                           ground_normal_threshold: float = 0.9,
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
        remove_ground (bool): Whether to remove the ground plane.
        remove_walls (bool): Whether to remove vertical walls.
        ground_normal_threshold (float): Threshold for ground plane detection.
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

    pcd_no_ground = pcd_filtered
    if remove_ground:
        logging.info("  -> Removing ground points...")
        plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=plane_dist_thresh,
                                                          ransac_n=3,
                                                          num_iterations=1000)
        a, b, c, _ = plane_model
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if abs(normal[2]) >= ground_normal_threshold:
            pcd_no_ground = pcd_filtered.select_by_index(inliers, invert=True)
            logging.info("     After ground removal: %d points", len(pcd_no_ground.points))
        else:
            logging.info("     Skipped ground removal: dominant plane is not ground-like")

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
                             max_correspondence_distance: float,
                             max_iteration: int,
                             method: int):
    """Perform iterative ICP fine registration.
       Parameters should be tuned according to the specific case.
    
    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud.
        target_pcd (o3d.geometry.PointCloud): Target point cloud.
        initial_transform (np.ndarray): Initial transformation matrix.
        max_correspondence_distance (float): ICP correspondence threshold.
        max_iteration (int): Maximum ICP iterations.

    Returns:
        o3d.pipelines.registration.RegistrationResult: Final ICP registration result.
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7,
        relative_rmse=1e-7,
        max_iteration=max_iteration)
    if method == 1:
        logging.info("  -> Using point-to-plane ICP")
        return o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, max_correspondence_distance, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria)
    if method == 2:
        logging.info("  -> Using GICP")
        return o3d.pipelines.registration.registration_generalized_icp(
            source_pcd, target_pcd, max_correspondence_distance, initial_transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria)
    logging.info("  -> Using point-to-point ICP")
    return o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria)


def build_refinement_stage_voxels(base_voxel_size: float) -> list[float]:
    """Build voxel sizes for multi-stage refinement from coarse to fine."""
    stage_voxels = []
    for scale in (1.0, 0.5, 0.25):
        voxel_size = max(base_voxel_size * scale, 0.01)
        voxel_size = round(voxel_size, 6)
        if voxel_size not in stage_voxels:
            stage_voxels.append(voxel_size)
    return stage_voxels


def perform_multistage_refinement(source_cloud: o3d.geometry.PointCloud,
                                  target_cloud: o3d.geometry.PointCloud,
                                  initial_transform: np.ndarray,
                                  preprocessing_params: dict,
                                  method: int):
    """Refine the extrinsic transform with a voxel pyramid."""
    current_transform = initial_transform
    final_result = None
    final_source_stage = None
    final_target_stage = None
    stage_voxels = build_refinement_stage_voxels(preprocessing_params['voxel_size'])

    for stage_index, voxel_size in enumerate(stage_voxels, start=1):
        logging.info("--- Refinement Stage %d/%d: voxel size %.3f m ---",
                     stage_index, len(stage_voxels), voxel_size)
        stage_params = dict(preprocessing_params)
        stage_params['voxel_size'] = voxel_size
        source_stage = preprocess_point_cloud(source_cloud, **stage_params)
        target_stage = preprocess_point_cloud(target_cloud, **stage_params)

        if len(source_stage.points) == 0 or len(target_stage.points) == 0:
            logging.error("Refinement stage %d produced an empty point cloud.", stage_index)
            return None, None, None

        max_correspondence_distance = max(voxel_size * 5, 0.02)
        max_iteration = 120 if stage_index == 1 else 80
        logging.info("  -> Max correspondence distance %.3f m", max_correspondence_distance)

        result = perform_icp_registration(
            source_stage,
            target_stage,
            current_transform,
            max_correspondence_distance,
            max_iteration,
            method,
        )
        current_transform = result.transformation
        final_result = result
        final_source_stage = source_stage
        final_target_stage = target_stage

    return final_result, final_source_stage, final_target_stage


def calibrate_lidar_extrinsic(source_cloud: o3d.geometry.PointCloud,
                              target_cloud: o3d.geometry.PointCloud,
                              is_draw_registration: bool = False,
                              preprocessing_params: dict = None,
                              method: int = 1,
                              initial_transform: np.ndarray | None = None):
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
            'remove_ground': False,
            'remove_walls': False
        }
    else:
        preprocessing_params = dict(preprocessing_params)
        preprocessing_params.setdefault('remove_ground', False)
        preprocessing_params.setdefault('remove_walls', False)
        preprocessing_params.setdefault('ground_normal_threshold', 0.9)
        preprocessing_params.setdefault('wall_angle_threshold', 0.1)
        preprocessing_params.setdefault('max_wall_planes', 2)

    logging.info("--- Step 1: Point Cloud Preprocessing ---")
    source_preprocessed = preprocess_point_cloud(source_cloud, **preprocessing_params)
    target_preprocessed = preprocess_point_cloud(target_cloud, **preprocessing_params)

    if len(source_preprocessed.points) == 0 or len(target_preprocessed.points) == 0:
        logging.error("Preprocessed point cloud is empty, calibration failed.")
        return None, None, None

    initial_guess = None
    if initial_transform is None:
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
    else:
        logging.info("--- Step 2: Skipping FPFH Feature Extraction ---")
        logging.info("--- Step 3: Using Provided Initial Transform ---")
        initial_guess_transform = np.asarray(initial_transform, dtype=float)
        logging.info("  - Initial transformation matrix:\n%s", initial_guess_transform)
        quat, trans = matrix_to_quaternion_and_translation(initial_guess_transform)
        logging.info("  - Rotation (quaternion wxyz): %s", quat)
        logging.info("  - Translation (xyz): %s", trans)

    if not np.isfinite(initial_guess_transform).all():
        logging.error("Coarse registration failed, invalid initial transformation matrix.")
        return None, None, None

    logging.info("--- Step 4: Fine Registration (Multi-stage Refinement) ---")
    registration_result, source_final_stage, target_final_stage = perform_multistage_refinement(
        source_cloud, target_cloud, initial_guess_transform, preprocessing_params, method)
    if registration_result is None:
        return None, initial_guess_transform, None
    final_extrinsic_transform = registration_result.transformation

    if not np.isfinite(final_extrinsic_transform).all():
        logging.error("Fine registration failed, invalid transformation matrix.")
        return None, initial_guess_transform, registration_result

    logging.info("\n--- Final Calibration Metrics ---")
    eval_params = dict(preprocessing_params)
    eval_params['voxel_size'] = build_refinement_stage_voxels(preprocessing_params['voxel_size'])[-1]
    source_eval = preprocess_point_cloud(source_cloud, **eval_params)
    target_eval = preprocess_point_cloud(target_cloud, **eval_params)
    source_points = np.asarray(source_eval.points)
    target_points = np.asarray(target_eval.points)
    source_points_h = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    source_points_transformed = (final_extrinsic_transform @ source_points_h.T).T[:, :3]
    tree = o3d.geometry.KDTreeFlann(target_eval)
    max_icp_corr_dist = 0.4
    matched_count = sum(
        1 for p in source_points_transformed
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
        if initial_guess is not None:
            logging.info("Visualizing coarse registration...")
            draw_registration_result(source_preprocessed, target_preprocessed,
                                     initial_guess_transform, window_name="Coarse Registration",
                                     registration_result=initial_guess)
        logging.info("Visualizing fine registration...")
        draw_registration_result(source_final_stage, target_final_stage,
                                 final_extrinsic_transform, window_name="Fine Registration",
                                 registration_result=registration_result,
                                 max_icp_corr_dist=max_icp_corr_dist)

    return final_extrinsic_transform, initial_guess_transform, registration_result

