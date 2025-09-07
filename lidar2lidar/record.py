# recorder.py
import open3d as o3d
import numpy as np
import os, sys, time, threading

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from cyber.python.cyber_py3 import cyber
from modules.common_msgs.sensor_msgs.pointcloud_pb2 import PointCloud

# --- 配置 ---
SAVE_FOLDER = "recorded_frames" # 保存点云的文件夹
FRAME_COUNT = 0
STOP_EVENT = threading.Event()

def pcd_callback(msg):
    global FRAME_COUNT

    print(f"\r正在保存第 {FRAME_COUNT} 帧...", end="")

    pcd = o3d.geometry.PointCloud()
    data_array = np.array([[p.x, p.y, p.z] for p in msg.point], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(data_array)

    if not pcd.has_points():
        print(f"\n警告：第 {FRAME_COUNT} 帧为空，已跳过。")
        return

    filename = os.path.join(SAVE_FOLDER, f"frame_{FRAME_COUNT:05d}.pcd")
    o3d.io.write_point_cloud(filename, pcd)
    FRAME_COUNT += 1

def user_input_listener(stop_event):
    input("\n[输入监听] 录制已开始。按【回车键】停止录制。\n")
    print("[输入监听] 已按下回车。正在停止录制...")
    stop_event.set()

if __name__ == '__main__':
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    cyber.init()
    recorder_node = cyber.Node("pcd_recorder_node")
    pcd_reader = recorder_node.create_reader(
        "/apollo/sensor/vanjeelidar/up/PointCloud2", PointCloud, pcd_callback)

    input_thread = threading.Thread(target=user_input_listener, args=(STOP_EVENT,), daemon=True)
    input_thread.start()

    print("="*50)
    print("      点云录制工具")
    print(f"数据将保存至: '{SAVE_FOLDER}' 文件夹")
    print("="*50)

    while not STOP_EVENT.is_set():
        time.sleep(0.1)

    print(f"\n录制结束。总共保存了 {FRAME_COUNT} 帧点云。")
    cyber.shutdown()