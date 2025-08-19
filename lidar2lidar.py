import open3d as o3d
import numpy as np
import logging

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
    """
    可视化配准结果。
    - 源点云（未变换）: 绿色
    - 目标点云: 蓝色
    - 变换后的源点云: 红色
    """
    source_temp = source.deepcopy()
    target_temp = target.deepcopy()
    source_temp.paint_uniform_color([0, 1, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def preprocess_point_cloud(pcd, voxel_size, nb_neighbors=20, std_ratio=2.0, plane_dist_thresh=0.05):
    """
    对点云进行预处理，包括降采样、离群点移除和地面移除。
    """
    logging.info("  -> 体素降采样...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    logging.info("  -> 移除统计离群点...")
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_filtered = pcd_down.select_by_index(ind)
    logging.info("  -> 移除地面点...")
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=plane_dist_thresh,
                                                      ransac_n=3,
                                                      num_iterations=1000)
    pcd_no_ground = pcd_filtered.select_by_index(inliers, invert=True)
    logging.info(f"  -> 原始点云: {len(pcd.points)} 点, 预处理后: {len(pcd_no_ground.points)} 点")
    logging.info("  -> 估计法线...")
    pcd_no_ground.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    return pcd_no_ground

def compute_fpfh_features(pcd, voxel_size):
    """
    计算点云的 FPFH 特征。
    """
    logging.info("  -> 计算 FPFH 特征...")
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_fpfh

def perform_coarse_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size):
    """
    使用 FPFH + RANSAC 进行鲁棒的初始对齐（粗配准）。
    """
    logging.info("  -> 执行粗配准 (FPFH + RANSAC)...")
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
    执行迭代式 GICP 精细配准。
    """
    current_transform = initial_transform
    final_result = None
    for i, max_corr_dist in enumerate(icp_params['max_correspondence_distances']):
        logging.info(f"  -> 迭代 {i+1}: 最大对应距离 {max_corr_dist} m")
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
    执行 LiDAR-to-LiDAR 外参标定的主函数。
    封装完整的标定流程，从预处理到最终的精细配准。
    """
    PREPROCESSING_PARAMS = {
        'voxel_size': 0.05,
        'nb_neighbors': 20,
        'std_ratio': 2.0,
        'plane_dist_thresh': 0.05
    }
    ICP_PARAMS = {
        'max_correspondence_distances': [1.0, 0.5, 0.25],
        'max_iterations': [50, 50, 50],
        'relative_fitness': 1e-7,
        'relative_rmse': 1e-7
    }

    logging.info("--- 步骤 1: 点云数据预处理 ---")
    source_preprocessed = preprocess_point_cloud(source_cloud, **PREPROCESSING_PARAMS)
    target_preprocessed = preprocess_point_cloud(target_cloud, **PREPROCESSING_PARAMS)

    logging.info("--- 步骤 2: FPFH 特征提取 ---")
    source_fpfh = compute_fpfh_features(source_preprocessed, PREPROCESSING_PARAMS['voxel_size'])
    target_fpfh = compute_fpfh_features(target_preprocessed, PREPROCESSING_PARAMS['voxel_size'])

    logging.info("--- 步骤 3: 粗配准 (FPFH + RANSAC) ---")
    initial_guess_transform = perform_coarse_registration(
        source_preprocessed, target_preprocessed, source_fpfh, target_fpfh, PREPROCESSING_PARAMS['voxel_size']
    )

    if not np.isfinite(initial_guess_transform).all():
        logging.error("粗配准失败，无法找到有效的初始变换矩阵。返回 None。")
        return None, None, None

    logging.info("--- 步骤 4: 精细配准 (迭代式 GICP) ---")
    registration_result = perform_icp_registration(
        source_preprocessed,
        target_preprocessed,
        initial_guess_transform,
        ICP_PARAMS
    )
    final_extrinsic_transform = registration_result.transformation

    logging.info("\n--- 最终标定指标 ---")
    logging.info(f"  - Fitness (重叠度): {registration_result.fitness:.4f}")
    logging.info(f"  - Inlier RMSE (内点均方根误差): {registration_result.inlier_rmse:.4f}")

    return final_extrinsic_transform, initial_guess_transform, registration_result

if __name__ == '__main__':
    logging.warning("这是一个作为库使用的文件，不应直接运行。请在你的主程序中导入并调用 'calibrate_lidar_extrinsic' 函数。")
