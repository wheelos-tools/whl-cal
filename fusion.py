import click
import numpy as np
import open3d as o3d
import pathlib
import yaml
from scipy.spatial.transform import Rotation as R
import subprocess
import tempfile
import os

def load_static_tf(filepath: pathlib.Path) -> np.ndarray:
    """Load a 4x4 extrinsic transformation matrix from YAML file."""
    with open(filepath, encoding='utf-8') as fin:
        extrinsics = yaml.load(fin, Loader=yaml.FullLoader)
        quat = (
            extrinsics['transform']['rotation']['x'],
            extrinsics['transform']['rotation']['y'],
            extrinsics['transform']['rotation']['z'],
            extrinsics['transform']['rotation']['w'],
        )
        translation = (
            extrinsics['transform']['translation']['x'],
            extrinsics['transform']['translation']['y'],
            extrinsics['transform']['translation']['z'],
        )
        rotation_matrix = R.from_quat(quat).as_matrix()
        tf = np.identity(4)
        tf[:3, :3] = rotation_matrix
        tf[:3, 3] = translation
        return tf


@click.command()
@click.option("--lf-pcd", type=pathlib.Path, required=True, help="Left front LiDAR PCD file")
@click.option("--rf-pcd", type=pathlib.Path, required=True, help="Right front LiDAR PCD file")
@click.option("--rb-pcd", type=pathlib.Path, required=True, help="Right back LiDAR PCD file")
@click.option("--lb-pcd", type=pathlib.Path, required=True, help="Left back LiDAR PCD file")
@click.option("--tf-dir", type=pathlib.Path, required=True, help="Directory containing YAML transform files")
@click.option("--use-pclview", is_flag=True, default=False, help="Use PCLview for visualization")
def main(lf_pcd, rf_pcd, rb_pcd, lb_pcd, tf_dir, use_pclview):
    """Visualize fused LiDAR point clouds in a single frame."""
    # Load static transforms
    lb2rb = load_static_tf(tf_dir / "left_back2right_back.yaml")
    rb2rf = load_static_tf(tf_dir / "right_back2right_front.yaml")
    rf2lf = load_static_tf(tf_dir / "right_front2left_front.yaml")

    # Load point clouds
    lf_pcd = o3d.io.read_point_cloud(str(lf_pcd))
    rf_pcd = o3d.io.read_point_cloud(str(rf_pcd))
    rb_pcd = o3d.io.read_point_cloud(str(rb_pcd))
    lb_pcd = o3d.io.read_point_cloud(str(lb_pcd))

    # Apply transformations and coloring
    lf_pcd.paint_uniform_color([1, 0, 0])  # Red
    rf_pcd.transform(rf2lf).paint_uniform_color([0, 1, 0])  # Green
    rb_pcd.transform(rb2rf).transform(rf2lf).paint_uniform_color([0, 0, 1])  # Blue
    lb_pcd.transform(lb2rb).transform(rb2rf).transform(rf2lf).paint_uniform_color([1, 1, 0])  # Yellow

    # Fuse all clouds
    fused_pcd = lf_pcd + rf_pcd + rb_pcd + lb_pcd

    if use_pclview:
        # Save to temporary PCD and visualize
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as tmpfile:
            temp_path = tmpfile.name
        o3d.io.write_point_cloud(temp_path, fused_pcd)
        subprocess.run(["pcl_viewer", temp_path])
        os.remove(temp_path)
    else:
        # Open3D visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Fused LiDAR Visualization")

        stop_flag = {"quit": False}

        def close_callback(v):
            v.close()
            return False

        vis.register_key_callback(ord("q"), close_callback)
        vis.register_key_callback(ord("Q"), close_callback)

        vis.add_geometry(fused_pcd)
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
