#!/usr/bin/env python3
"""extract_pcd.py
"""

import datetime
import os
import click
from cyber_record.record import Record
from record_msg.parser import PointCloudParser


@click.command()
@click.option('--input_dir',
              '-i',
              type=click.Path(exists=True, dir_okay=True),
              required=True,
              help='Path to the input.')
@click.option('--output_dir',
              '-o',
              type=click.Path(exists=True, dir_okay=True),
              required=True,
              help='Path to the output.')
@click.option('--channels',
              '-c',
              type=str,
              default=[
                  '/apollo/sensor/vanjeelidar/left_front/PointCloud2',
                  '/apollo/sensor/vanjeelidar/left_back/PointCloud2',
                  '/apollo/sensor/vanjeelidar/right_front/PointCloud2',
                  '/apollo/sensor/vanjeelidar/right_back/PointCloud2',
                  '/apollo/sensor/lidar/fusion/PointCloud2',
              ],
              multiple=True,
              help='Channel to extract point cloud messages from.')
def main(input_dir: str, output_dir: str, channels):
    """extract_pcd
    """
    files = [
        f for f in [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
        if os.path.isfile(f)
    ]
    os.makedirs(output_dir, exist_ok=True)
    pointcloud_parser = PointCloudParser(output_dir, True, '.pcd')
    for file in files:
        with Record(file) as record:
            for channel, msg, t in record.read_messages(topics=channels):
                time_str = datetime.datetime.fromtimestamp(
                    t / 1e9).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                sensor_name = channel.split('/')[-2]
                os.makedirs(os.path.join(output_dir, sensor_name),
                            exist_ok=True)
                pointcloud_parser.parse(msg,
                                        file_name=f'{sensor_name}/{time_str}',
                                        mode='ascii')


if __name__ == '__main__':
    main()
