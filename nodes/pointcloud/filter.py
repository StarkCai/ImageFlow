"""点云滤波算子：统计滤波 / 半径滤波 / 中值滤波 / 均值滤波 / 双边滤波 / 直通滤波。"""

from __future__ import annotations
import numpy as np

from node_base import Node
from node_registry import register_node


def _apply_statistical(pcd, nb_neighbors: int, std_ratio: float):
    import open3d as o3d
    result, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return result


def _apply_radius(pcd, nb_points: int, radius: float):
    import open3d as o3d
    result, _ = pcd.remove_radius_outlier(nb_points, radius)
    return result


def _apply_median(pcd, kernel_size: int):
    """KDTree 邻域中值滤波。"""
    import open3d as o3d
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    filtered = np.zeros_like(points)
    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], kernel_size + 1)
        filtered[i] = np.median(points[idx], axis=0)

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(filtered)
    if pcd.has_colors():
        result.colors = pcd.colors
    if pcd.has_normals():
        result.normals = pcd.normals
    return result


def _apply_mean(pcd, kernel_size: int):
    """KDTree 邻域均值滤波。"""
    import open3d as o3d
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    filtered = np.zeros_like(points)
    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], kernel_size + 1)
        filtered[i] = np.mean(points[idx], axis=0)

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(filtered)
    if pcd.has_colors():
        result.colors = pcd.colors
    if pcd.has_normals():
        result.normals = pcd.normals
    return result


def _apply_bilateral(pcd, nb_neighbors: int, sigma_s: float, sigma_r: float):
    """点云双边滤波：空间权重 + 强度权重。"""
    import open3d as o3d
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    filtered = np.zeros_like(points)
    for i in range(len(points)):
        _, idx, dists = tree.search_knn_vector_3d(pcd.points[i], nb_neighbors)
        dists = np.sqrt(dists)
        # 空间权重
        w_s = np.exp(-0.5 * (dists / sigma_s) ** 2)
        # 强度权重（深度相对偏移）
        depth_self = points[i, 2]
        depth_neighbors = points[idx, 2]
        w_r = np.exp(-0.5 * ((depth_neighbors - depth_self) / sigma_r) ** 2)
        w = w_s * w_r
        w = np.maximum(w, 1e-10)
        filtered[i] = np.sum(points[idx] * w[:, None], axis=0) / w.sum()

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(filtered)
    if pcd.has_colors():
        result.colors = pcd.colors
    if pcd.has_normals():
        result.normals = pcd.normals
    return result


def _apply_passthrough(pcd, axis: str, min_val: float, max_val: float):
    """直通滤波：沿指定轴截取范围。"""
    points = np.asarray(pcd.points)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    mask = (points[:, axis_idx] >= min_val) & (points[:, axis_idx] <= max_val)
    return pcd.select_by_index(np.where(mask)[0])


@register_node
class PointCloudFilterNode(Node):
    display_name = "点云滤波"
    category = "点云处理"
    algorithms = ["统计滤波", "半径滤波", "中值滤波", "均值滤波", "双边滤波", "直通滤波"]

    def __init__(self):
        self.algorithm: str = "统计滤波"
        # 统计/半径/双边共用
        self.nb_neighbors: int = 20
        # 统计滤波
        self.std_ratio: float = 2.0
        # 半径滤波
        self.radius: float = 0.1
        # 中值/均值
        self.kernel_size: int = 10
        # 双边滤波
        self.sigma_s: float = 0.1
        self.sigma_r: float = 0.1
        # 直通滤波
        self.axis: str = "z"
        self.min_val: float = -1.0
        self.max_val: float = 1.0
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        if self.algorithm == "统计滤波":
            return {"点云": _apply_statistical(pcd, self.nb_neighbors, self.std_ratio)}
        elif self.algorithm == "半径滤波":
            return {"点云": _apply_radius(pcd, self.nb_neighbors, self.radius)}
        elif self.algorithm == "中值滤波":
            return {"点云": _apply_median(pcd, self.kernel_size)}
        elif self.algorithm == "均值滤波":
            return {"点云": _apply_mean(pcd, self.kernel_size)}
        elif self.algorithm == "双边滤波":
            return {"点云": _apply_bilateral(pcd, self.nb_neighbors, self.sigma_s, self.sigma_r)}
        elif self.algorithm == "直通滤波":
            return {"点云": _apply_passthrough(pcd, self.axis, self.min_val, self.max_val)}
        return {"点云": pcd}
