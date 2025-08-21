import click
import numpy as np
import open3d
import pathlib
import copy
import yaml
from scipy.spatial.transform import Rotation as R


def load_static_tf(filepath):
    """load_static_tf
  """
    with open(filepath, encoding='utf-8') as fin:
        extrinsics = yaml.load(fin, Loader=yaml.FullLoader)
        orientation = (
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
        rotation_matrix = R.from_quat(orientation).as_matrix()
        tf = np.identity(4)
        tf[:3, :3] = rotation_matrix
        tf[:3, 3] = translation
        return tf


@click.command()
def main():
    """fusion
  """
    prefix = pathlib.Path('/home/foliage/ring/'
                          'github.com/wheelos-tools/apollo-zhongji/'
                          'modules/drivers/lidar/params')

    lf2rf = load_static_tf(prefix / 'lf_to_rf_extrinsics.yaml')
    rf2rr = load_static_tf(prefix / 'rf_to_rr_extrinsics.yaml')
    rr2lr = load_static_tf(prefix / 'rr_to_lr_extrinsics.yaml')
    print('lf2rf', lf2rf)

    lf_pcd = open3d.io.read_point_cloud(
        'cal_5_pcds/left_front/2025-08-20_18-58-43-622.pcd', format='pcd')
    rf_pcd = open3d.io.read_point_cloud(
        'cal_5_pcds/right_front/2025-08-20_18-58-43-580.pcd', format='pcd')
    rr_pcd = open3d.io.read_point_cloud(
        'cal_5_pcds/right_back/2025-08-20_18-58-43-575.pcd', format='pcd')
    lr_pcd = open3d.io.read_point_cloud(
        'cal_5_pcds/left_back/2025-08-20_18-58-43-565.pcd', format='pcd')

    lf_pcd_tp = copy.deepcopy(lf_pcd)
    rf_pcd_tp = copy.deepcopy(rf_pcd)
    rr_pcd_tp = copy.deepcopy(rr_pcd)
    lr_pcd_tp = copy.deepcopy(lr_pcd)

    lf_pcd_tp.paint_uniform_color([1, 0, 0])  # Red

    rf_pcd_tp.transform(lf2rf)
    rf_pcd_tp.paint_uniform_color([0, 1, 0])  # Green

    rr_pcd_tp.transform(rf2rr).transform(lf2rf)
    rr_pcd_tp.paint_uniform_color([0, 0, 1])  # Blue

    lr_pcd_tp.transform(rr2lr).transform(rf2rr).transform(lf2rf)
    lr_pcd_tp.paint_uniform_color([1, 1, 0])  # Yellow

    open3d.visualization.draw_geometries(
        [lf_pcd_tp, rf_pcd_tp, rr_pcd_tp, lr_pcd_tp],
        window_name='Fusion Result')


if __name__ == '__main__':
    main()
