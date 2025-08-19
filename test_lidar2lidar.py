import open3d as o3d
import numpy as np
import logging
import copy
import lidar2lidar as lc

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_test_data():
    """
    Generates mock point cloud data for testing.
    Returns:
        tuple: (source_cloud, target_cloud, ground_truth_transform)
    """
    logging.info("1. Generating mock point cloud data...")
    box_points = []
    # Create an L-shaped wall structure for rich features
    for _ in range(2000):
        box_points.append([np.random.uniform(0, 2), np.random.uniform(0, 0.1), np.random.uniform(0, 1.5)])
        box_points.append([np.random.uniform(0, 0.1), np.random.uniform(0, 2), np.random.uniform(0, 1.5)])
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(np.array(box_points, dtype=np.float64))

    # Define the "ground truth" extrinsic (the value we want to find)
    true_rotation = target_cloud.get_rotation_matrix_from_xyz((0, 0, np.pi / 12))  # 15 degrees
    true_translation = np.array([0.5, -0.3, 0.2])
    ground_truth_transform = np.identity(4)
    ground_truth_transform[0:3, 0:3] = true_rotation
    ground_truth_transform[0:3, 3] = true_translation

    # Create the source point cloud by transforming the target
    source_cloud = copy.deepcopy(target_cloud)
    source_cloud.transform(ground_truth_transform)

    return source_cloud, target_cloud, ground_truth_transform

def main_test():
    """
    Main function to run the calibration test case.
    """
    # Load or generate test data
    source_cloud, target_cloud, ground_truth_transform = generate_test_data()

    # --- Execute the calibration ---
    logging.info("\n--- Starting LiDAR Extrinsic Calibration ---")
    final_extrinsic_transform, coarse_transform, reg_result = lc.calibrate_lidar_extrinsic(source_cloud, target_cloud)

    if final_extrinsic_transform is None:
        logging.error("Calibration failed.")
        return

    # --- Result validation and visualization ---
    logging.info("\n--- Final Calibration Results ---")
    logging.info(f"Ground truth extrinsic matrix:\n{np.round(ground_truth_transform, 4)}")
    logging.info(f"Coarse registration extrinsic matrix:\n{np.round(coarse_transform, 4)}")
    logging.info(f"Computed final extrinsic matrix:\n{np.round(final_extrinsic_transform, 4)}")

    # Visualize results
    logging.info("Visualizing final registration results.")
    # Note: We visualize with the original clouds for best quality
    lc.draw_registration_result(source_cloud, target_cloud, final_extrinsic_transform, "Final Fine Registration Result")

    logging.info("Test completed successfully.")

if __name__ == '__main__':
    main_test()
