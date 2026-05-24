"""点云读取算子：支持单文件读取（PLY/PCD/XYZ/PTS等格式）。"""

from __future__ import annotations
import os

import numpy as np

from node_base import Node
from node_registry import register_node


def _read_pointcloud(filepath: str):
    """读取点云文件，返回 open3d.geometry.PointCloud。"""
    import open3d as o3d

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xyz", ".pts", ".txt", ".xyzn", ".xyzrgb"):
        data = np.loadtxt(filepath, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"点云文件格式不正确: {filepath}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0
                                                     if data[:, 3:6].max() > 1.0
                                                     else data[:, 3:6])
        return pcd

    pcd = o3d.io.read_point_cloud(filepath)
    if pcd is None or not pcd.has_points():
        raise ValueError(f"无法读取点云文件: {filepath}")
    return pcd


@register_node
class PointCloudInputNode(Node):
    display_name = "点云读取"
    category = "输入"

    def __init__(self):
        self.file_path: str = ""
        super().__init__()

    def _setup_ports(self):
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        if not self.file_path:
            raise ValueError("未设置点云文件路径")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        pcd = _read_pointcloud(self.file_path)
        return {"点云": pcd}
