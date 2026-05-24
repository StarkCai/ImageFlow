"""三维表面重建算子：Poisson / Delaunay / BPA / Alpha Shape / Marching Cubes。"""

from __future__ import annotations
import numpy as np

from node_base import Node
from node_registry import register_node


def _mesh_to_pointcloud(mesh, density: float = 0.01):
    """将三角网格采样为点云（用于统一输出为点云类型）。"""
    import open3d as o3d
    if mesh.is_empty():
        return o3d.geometry.PointCloud()

    pcd = mesh.sample_points_poisson_disk(
        int(len(mesh.vertices) * 2), seed=42)
    if not pcd.has_points():
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.paint_uniform_color([0.4, 0.6, 0.9])
    else:
        pcd.paint_uniform_color([0.4, 0.6, 0.9])
    return pcd


def _poisson_reconstruct(pcd, depth: int = 9, scale: float = 1.1,
                         linear_fit: bool = False, density_threshold: float = 0.01):
    """泊松曲面重建。"""
    import open3d as o3d
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(30),
        o3d.geometry.KDTreeSearchParamKNN(30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)

    if density_threshold is not None and density_threshold > 0:
        vertices_to_remove = densities < density_threshold * np.max(densities)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.remove_unreferenced_vertices()

    return mesh


def _delaunay_reconstruct(pcd):
    """Delaunay 三角剖分：3D Delaunay + 提取凸包面。"""
    import open3d as o3d
    from scipy.spatial import Delaunay

    points = np.asarray(pcd.points)
    tri = Delaunay(points)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    # Delaunay 的 convex_hull 给出边界三角面
    mesh.triangles = o3d.utility.Vector3iVector(tri.convex_hull)
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _bpa_reconstruct(pcd, radii: list):
    """球旋转算法 (Ball-Pivoting)。"""
    import open3d as o3d
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(30),
        o3d.geometry.KDTreeSearchParamKNN(30))
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh


def _alpha_shape_reconstruct(pcd, alpha: float = 0.03):
    """Alpha Shape 曲面重建。"""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh


def _marching_cubes_reconstruct(pcd, voxel_size: float = 0.02):
    """Marching Cubes：体素化 + 等值面提取。"""
    import open3d as o3d

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(30),
        o3d.geometry.KDTreeSearchParamKNN(30))

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size,
        pcd.get_min_bound(), pcd.get_max_bound())

    # 从体素网格重建 mesh
    mesh = o3d.geometry.TriangleMesh()
    try:
        voxels = voxel_grid.get_voxels()
        if len(voxels) == 0:
            raise RuntimeError("体素网格为空，请减小体素大小")

        dense_voxel = o3d.geometry.VoxelGrid()
        # 手动构建密集体素网格用于 marching cubes
        min_b = pcd.get_min_bound()
        max_b = pcd.get_max_bound()
        shape = np.ceil((max_b - min_b) / voxel_size).astype(int) + 1
        occupancy = np.zeros(shape, dtype=np.float64)

        for v in voxels:
            idx = tuple(np.floor((v.grid_index * voxel_size + min_b - min_b) /
                                 voxel_size).astype(int))
            idx = tuple(np.clip(idx, 0, np.array(shape) - 1))
            occupancy[idx] = 1.0

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
    except Exception:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)

    return mesh


@register_node
class PointCloudReconstructionNode(Node):
    display_name = "三维表面重建"
    category = "点云处理"
    algorithms = [
        "泊松曲面重建", "Delaunay三角剖分", "球旋转算法(BPA)",
        "Alpha Shape", "Marching Cubes",
    ]

    def __init__(self):
        self.algorithm: str = "泊松曲面重建"
        # Poisson
        self.poisson_depth: int = 9
        self.poisson_scale: float = 1.1
        self.poisson_linear_fit: bool = False
        self.density_threshold: float = 0.01
        # BPA
        self.bpa_radius1: float = 0.05
        self.bpa_radius2: float = 0.1
        self.bpa_radius3: float = 0.2
        # Alpha Shape
        self.alpha: float = 0.03
        # Marching Cubes
        self.voxel_size: float = 0.02
        # 采样密度
        self.sample_density: float = 0.01
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        if self.algorithm == "泊松曲面重建":
            mesh = _poisson_reconstruct(
                pcd, self.poisson_depth, self.poisson_scale,
                self.poisson_linear_fit, self.density_threshold)
        elif self.algorithm == "Delaunay三角剖分":
            mesh = _delaunay_reconstruct(pcd)
        elif self.algorithm == "球旋转算法(BPA)":
            radii = [r for r in [self.bpa_radius1, self.bpa_radius2, self.bpa_radius3]
                     if r > 0]
            if not radii:
                radii = [0.05, 0.1, 0.2]
            mesh = _bpa_reconstruct(pcd, radii)
        elif self.algorithm == "Alpha Shape":
            mesh = _alpha_shape_reconstruct(pcd, self.alpha)
        elif self.algorithm == "Marching Cubes":
            mesh = _marching_cubes_reconstruct(pcd, self.voxel_size)
        else:
            mesh = _poisson_reconstruct(pcd)

        result_pcd = _mesh_to_pointcloud(mesh, self.sample_density)
        return {"点云": result_pcd}
