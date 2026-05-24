"""点云分割算子：RANSAC / 区域生长分割 / 超体聚类。"""

from __future__ import annotations
import numpy as np

from node_base import Node
from node_registry import register_node


def _ransac_segment(pcd, distance_threshold: float, ransac_n: int, num_iterations: int):
    """RANSAC 平面分割，返回内点着色点云。"""
    import open3d as o3d
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])
    return inlier_cloud


def _estimate_normals(points: np.ndarray, nb_neighbors: int = 30):
    """基于 PCA 估计点云法线。返回 (normals, curvatures)。"""
    import open3d as o3d
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd_tmp)

    normals = np.zeros_like(points)
    curvatures = np.zeros(len(points))

    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(pcd_tmp.points[i], nb_neighbors)
        neighbors = points[idx]
        centroid = neighbors.mean(axis=0)
        cov = (neighbors - centroid).T @ (neighbors - centroid) / len(neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]  # 最小特征值对应的向量
        curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-10)

    return normals, curvatures


def _region_growing(pcd, nb_neighbors: int, smoothness_threshold: float,
                    curvature_threshold: float, min_cluster_size: int):
    """区域生长分割：基于法线和曲率。"""
    import open3d as o3d
    points = np.asarray(pcd.points).copy()
    n = len(points)

    normals, curvatures = _estimate_normals(points, nb_neighbors)

    # KDTree 用于邻域搜索
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd_tmp)

    # 标记：-1 = 未分类，>=0 = 聚类 ID
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    # 按曲率升序排列种子点（曲率小的点优先）
    seed_order = np.argsort(curvatures)

    for seed_idx in seed_order:
        if labels[seed_idx] >= 0:
            continue
        if curvatures[seed_idx] > curvature_threshold:
            continue

        # 初始化新区域
        labels[seed_idx] = cluster_id
        region = [seed_idx]

        while region:
            idx = region.pop(0)
            normal_i = normals[idx]

            # 查找邻域
            _, neighbors, _ = tree.search_knn_vector_3d(pcd_tmp.points[idx], nb_neighbors)

            for nb in neighbors:
                if labels[nb] >= 0:
                    continue
                # 法线角度差
                dot = abs(np.dot(normal_i, normals[nb]))
                if dot < np.cos(np.radians(smoothness_threshold * 90)):
                    continue
                # 曲率检查
                if curvatures[nb] > curvature_threshold:
                    continue
                labels[nb] = cluster_id
                region.append(nb)

        if np.sum(labels == cluster_id) >= min_cluster_size:
            cluster_id += 1
        else:
            labels[labels == cluster_id] = -1

    # 着色
    max_label = labels.max()
    np.random.seed(42)
    colors_arr = np.random.rand(max(1, max_label + 1), 3)
    colors_arr[0] = [0, 0, 0]  # 噪声为黑色

    out_colors = np.zeros((n, 3))
    for i in range(n):
        lbl = labels[i]
        out_colors[i] = colors_arr[lbl] if lbl >= 0 else [0, 0, 0]

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(out_colors)
    return result


def _supervoxel_clustering(pcd, voxel_size: float, seed_resolution: float,
                           color_importance: float, spatial_importance: float):
    """超体聚类：基于体素化的颜色-空间联合聚类。"""
    import open3d as o3d
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((len(points), 3))
    n = len(points)

    # 体素化
    pmin = points.min(axis=0)
    voxel_indices = np.floor((points - pmin) / voxel_size).astype(np.int64)
    voxel_keys, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    n_voxels = len(voxel_keys)

    # 每个体素计算中心点和平均颜色
    voxel_centers = np.zeros((n_voxels, 3))
    voxel_colors = np.zeros((n_voxels, 3))
    for v in range(n_voxels):
        mask = inv == v
        voxel_centers[v] = points[mask].mean(axis=0)
        voxel_colors[v] = colors[mask].mean(axis=0)

    # 在体素中心上做 K-means 聚类
    n_clusters = max(1, int(seed_resolution * n_voxels))
    n_clusters = max(2, min(n_clusters, n_voxels))

    # 简单 K-means
    features = np.column_stack([
        voxel_centers * spatial_importance,
        voxel_colors * color_importance * 0.5,
    ])

    np.random.seed(42)
    idx = np.random.choice(n_voxels, n_clusters, replace=False)
    centers = features[idx].copy()

    for _ in range(30):
        dists = np.array([((features - c) ** 2).sum(axis=1) for c in centers])
        assignments = np.argmin(dists, axis=0)
        new_centers = np.array([
            features[assignments == k].mean(axis=0)
            if (assignments == k).sum() > 0 else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centers, new_centers, rtol=1e-4):
            break
        centers = new_centers

    # 将体素标签映射回点
    point_labels = assignments[inv]

    # 着色
    cmap = np.random.RandomState(42).rand(n_clusters, 3)
    out_colors = cmap[point_labels]

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(out_colors)
    return result


@register_node
class PointCloudSegmentationNode(Node):
    display_name = "点云分割"
    category = "点云处理"
    algorithms = ["RANSAC", "区域生长分割", "超体聚类"]

    def __init__(self):
        self.algorithm: str = "RANSAC"
        # RANSAC
        self.distance_threshold: float = 0.01
        self.ransac_n: int = 3
        self.num_iterations: int = 1000
        # 区域生长
        self.nb_neighbors: int = 30
        self.smoothness_threshold: float = 0.05
        self.curvature_threshold: float = 0.05
        self.min_cluster_size: int = 50
        # 超体聚类
        self.voxel_size: float = 0.05
        self.seed_resolution: float = 0.1
        self.color_importance: float = 0.2
        self.spatial_importance: float = 0.4
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        if self.algorithm == "RANSAC":
            return {"点云": _ransac_segment(
                pcd, self.distance_threshold, self.ransac_n, self.num_iterations)}
        elif self.algorithm == "区域生长分割":
            return {"点云": _region_growing(
                pcd, self.nb_neighbors, self.smoothness_threshold,
                self.curvature_threshold, self.min_cluster_size)}
        elif self.algorithm == "超体聚类":
            return {"点云": _supervoxel_clustering(
                pcd, self.voxel_size, self.seed_resolution,
                self.color_importance, self.spatial_importance)}
        return {"点云": pcd}
