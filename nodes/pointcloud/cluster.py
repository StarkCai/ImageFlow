"""点云聚类算子：DBSCAN / K-means / 欧式聚类 / 均值漂移聚类。"""

from __future__ import annotations
import numpy as np

from node_base import Node
from node_registry import register_node


def _dbscan_cluster(pcd, eps: float, min_points: int):
    """DBSCAN 聚类，返回着色点云。"""
    import open3d as o3d
    labels = np.asarray(pcd.cluster_dbscan(eps, min_points, print_progress=False))

    max_label = labels.max()
    np.random.seed(42)
    colors = np.random.rand(max_label + 1, 3)
    colors[0] = [0, 0, 0]  # 噪声为黑色

    colored = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        colored[i] = colors[labels[i]] if labels[i] >= 0 else [0, 0, 0]

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(colored)
    return result


def _kmeans_cluster(pcd, n_clusters: int):
    """K-means 聚类，返回着色点云。"""
    import open3d as o3d
    points = np.asarray(pcd.points)

    # K-means++ 初始化
    np.random.seed(42)
    centers = [points[np.random.randint(len(points))]]
    for _ in range(1, n_clusters):
        dists = np.min([np.sum((points - c) ** 2, axis=1) for c in centers], axis=0)
        probs = dists / dists.sum()
        centers.append(points[np.random.choice(len(points), p=probs)])

    centers = np.array(centers)

    for _ in range(100):
        dists = np.array([np.sum((points - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        new_centers = np.array([
            points[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    np.random.seed(42)
    cmap = np.random.rand(n_clusters, 3)
    colored = cmap[labels]

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(colored)
    return result


def _euclidean_cluster(pcd, eps: float, min_points: int):
    """欧式聚类：与 DBSCAN 类似但基于欧式距离阈值进行分组。"""
    import open3d as o3d
    # open3d 没有直接提供欧式聚类，用 DBSCAN 近似
    labels = np.asarray(pcd.cluster_dbscan(eps, min_points, print_progress=False))

    max_label = labels.max()
    np.random.seed(42)
    colors = np.random.rand(max_label + 1, 3)
    colors[0] = [0, 0, 0]

    colored = np.zeros((len(labels), 3))
    for i in range(len(labels)):
        colored[i] = colors[labels[i]] if labels[i] >= 0 else [0, 0, 0]

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(colored)
    return result


def _mean_shift_cluster(pcd, bandwidth: float, max_iter: int):
    """均值漂移聚类，返回着色点云。"""
    points = np.asarray(pcd.points).copy()
    n = len(points)
    shifted = points.copy()

    for _ in range(max_iter):
        for i in range(n):
            dists = np.sum((points - shifted[i]) ** 2, axis=1)
            in_band = dists < bandwidth ** 2
            if in_band.sum() > 0:
                shifted[i] = points[in_band].mean(axis=0)

    # 将收敛到同一中心的点归为一类
    centers = []
    labels = np.full(n, -1, dtype=int)
    for i in range(n):
        assigned = False
        for c_idx, c in enumerate(centers):
            if np.sum((shifted[i] - c) ** 2) < (bandwidth * 0.5) ** 2:
                labels[i] = c_idx
                assigned = True
                break
        if not assigned:
            labels[i] = len(centers)
            centers.append(shifted[i])

    np.random.seed(42)
    n_clusters = len(centers)
    cmap = np.random.rand(max(n_clusters, 1), 3)
    if n_clusters == 0:
        cmap = np.array([[0, 0, 0]])

    colored = np.zeros((n, 3))
    for i in range(n):
        colored[i] = cmap[labels[i]] if labels[i] >= 0 else [0, 0, 0]

    import open3d as o3d
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(colored)
    return result


@register_node
class PointCloudClusterNode(Node):
    display_name = "点云聚类"
    category = "点云处理"
    algorithms = ["DBSCAN", "K-means", "欧式聚类", "均值漂移聚类"]

    def __init__(self):
        self.algorithm: str = "DBSCAN"
        # DBSCAN / 欧式聚类
        self.eps: float = 0.05
        self.min_points: int = 10
        # K-means
        self.n_clusters: int = 5
        # 均值漂移
        self.bandwidth: float = 0.1
        self.max_iter: int = 50
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        if self.algorithm == "DBSCAN":
            return {"点云": _dbscan_cluster(pcd, self.eps, self.min_points)}
        elif self.algorithm == "K-means":
            return {"点云": _kmeans_cluster(pcd, self.n_clusters)}
        elif self.algorithm == "欧式聚类":
            return {"点云": _euclidean_cluster(pcd, self.eps, self.min_points)}
        elif self.algorithm == "均值漂移聚类":
            return {"点云": _mean_shift_cluster(pcd, self.bandwidth, self.max_iter)}
        return {"点云": pcd}
