"""点云配准算子：支持多输入点云配准（粗配准/精配准），输出评估指标。"""

from __future__ import annotations
import numpy as np

from node_base import Node
from node_registry import register_node


def _pca_align(source, target):
    """PCA 粗配准：对齐主成分轴。"""
    src_c = source - source.mean(axis=0)
    tgt_c = target - target.mean(axis=0)

    _, _, vt_src = np.linalg.svd(src_c, full_matrices=False)
    _, _, vt_tgt = np.linalg.svd(tgt_c, full_matrices=False)

    R = vt_tgt.T @ vt_src
    if np.linalg.det(R) < 0:
        vt_tgt[-1] *= -1
        R = vt_tgt.T @ vt_src

    t = target.mean(axis=0) - R @ source.mean(axis=0)
    return R, t


def _cpd_align(source, target, max_iter: int = 50, tol: float = 1e-4,
               w: float = 0.0):
    """CPD (Coherent Point Drift) 刚性配准。"""
    n, m = len(source), len(target)
    R = np.eye(3)
    t = np.zeros(3)
    sigma2 = np.sum((source - target.mean(axis=0)) ** 2) / (3 * n)

    Y = source.copy()
    for _ in range(max_iter):
        # E-step: 计算后验概率
        diff = target[None, :, :] - Y[:, None, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        P = np.exp(-sq_dist / (2 * sigma2))
        P_sum = P.sum(axis=1) + w / (1 - w) * n * (2 * np.pi * sigma2) ** 1.5
        P = P / P_sum[:, None]

        # M-step: 求解刚性变换
        mu_t = P.sum(axis=1)
        mu_s = P.sum(axis=0)
        T_hat = target - mu_t[:, None] * Y
        S_hat = source - mu_s[:, None] * Y

        U, _, Vt = np.linalg.svd(S_hat.T @ T_hat)
        new_R = Vt.T @ U.T
        if np.linalg.det(new_R) < 0:
            Vt[-1] *= -1
            new_R = Vt.T @ U.T
        new_t = (mu_t - new_R @ mu_s) / n

        Y_new = (new_R @ source.T).T + new_t
        new_sigma2 = (np.sum(np.sum(P * sq_dist, axis=1)) /
                      (3 * P.sum()))
        new_sigma2 = max(new_sigma2, 1e-10)

        if (np.linalg.norm(new_R - R) < tol and
                np.linalg.norm(new_t - t) < tol and
                abs(new_sigma2 - sigma2) / sigma2 < tol):
            break

        R, t, sigma2, Y = new_R, new_t, new_sigma2, Y_new

    return R, t


def _do_coarse_registration(source_pcd, target_pcd, algorithm: str, params: dict):
    """粗配准入口。返回 (aligned_pcd, transformation_4x4)。"""
    import open3d as o3d
    reg = o3d.pipelines.registration

    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)

    if algorithm == "FPFH+RANSAC配准":
        voxel = params.get("voxel_size", 0.05)
        src_down = source_pcd.voxel_down_sample(voxel)
        tgt_down = target_pcd.voxel_down_sample(voxel)
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        src_fpfh = reg.compute_fpfh_feature(src_down, o3d.geometry.KDTreeSearchParamKNN(100))
        tgt_fpfh = reg.compute_fpfh_feature(tgt_down, o3d.geometry.KDTreeSearchParamKNN(100))
        result = reg.registration_ransac_based_on_feature_matching(
            src_down, tgt_down, src_fpfh, tgt_fpfh, True,
            params.get("distance_threshold", 0.05),
            reg.TransformationEstimationPointToPoint(), params.get("ransac_n", 3),
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             reg.CorrespondenceCheckerBasedOnDistance(params.get("distance_threshold", 0.05))],
            reg.RANSACConvergenceCriteria(params.get("max_iterations", 100000), 0.999))

    elif algorithm == "快速全局配准(FGR)":
        voxel = params.get("voxel_size", 0.05)
        src_down = source_pcd.voxel_down_sample(voxel)
        tgt_down = target_pcd.voxel_down_sample(voxel)
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        src_fpfh = reg.compute_fpfh_feature(src_down, o3d.geometry.KDTreeSearchParamKNN(100))
        tgt_fpfh = reg.compute_fpfh_feature(tgt_down, o3d.geometry.KDTreeSearchParamKNN(100))
        result = reg.registration_fgr_based_on_feature_matching(
            src_down, tgt_down, src_fpfh, tgt_fpfh,
            reg.FastGlobalRegistrationOption(
                division_factor=params.get("division_factor", 1.4),
                decrease_mu=params.get("decrease_mu", True),
                maximum_correspondence_distance=params.get("distance_threshold", 0.05),
                iteration_number=params.get("max_iterations", 64),
            ))

    elif algorithm == "CPD配准":
        R, t_vec = _cpd_align(src_pts, tgt_pts,
                              params.get("max_iterations", 50),
                              params.get("cpd_tol", 1e-4))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t_vec
        result = reg.RegistrationResult()
        result.transformation = T
        result.fitness = 0.0
        result.inlier_rmse = 0.0

    elif algorithm == "PCA配准":
        R, t_vec = _pca_align(src_pts, tgt_pts)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t_vec
        result = reg.RegistrationResult()
        result.transformation = T
        result.fitness = 0.0
        result.inlier_rmse = 0.0

    else:
        raise ValueError(f"未知粗配准算法: {algorithm}")

    aligned = source_pcd.transform(result.transformation)
    return aligned, result


def _do_fine_registration(source_pcd, target_pcd, algorithm: str, params: dict,
                          init_transformation=None):
    """精配准入口。返回 (aligned_pcd, registration_result)。"""
    import open3d as o3d
    reg = o3d.pipelines.registration

    threshold = params.get("distance_threshold", 0.02)
    max_iter = params.get("max_iterations", 2000)
    criteria = reg.ICPConvergenceCriteria(
        relative_fitness=params.get("relative_fitness", 1e-6),
        relative_rmse=params.get("relative_rmse", 1e-6),
        max_iteration=max_iter)

    init = (np.eye(4) if init_transformation is None
            else init_transformation)

    if algorithm == "点对点ICP":
        result = reg.registration_icp(
            source_pcd, target_pcd, threshold, init,
            reg.TransformationEstimationPointToPoint(), criteria)
    elif algorithm == "点对面ICP":
        source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        result = reg.registration_icp(
            source_pcd, target_pcd, threshold, init,
            reg.TransformationEstimationPointToPlane(), criteria)
    elif algorithm == "广义ICP(GICP)":
        result = reg.registration_generalized_icp(
            source_pcd, target_pcd, threshold, init,
            reg.TransformationEstimationForGeneralizedICP(), criteria)
    elif algorithm == "彩色ICP":
        result = reg.registration_colored_icp(
            source_pcd, target_pcd, threshold, init,
            reg.ICPConvergenceCriteria(
                relative_fitness=params.get("relative_fitness", 1e-6),
                relative_rmse=params.get("relative_rmse", 1e-6),
                max_iteration=max_iter))
    else:
        raise ValueError(f"未知精配准算法: {algorithm}")

    aligned = source_pcd.transform(result.transformation)
    return aligned, result


@register_node
class PointCloudRegistrationNode(Node):
    display_name = "点云配准"
    category = "点云处理"
    algorithms = [
        "FPFH+RANSAC配准", "快速全局配准(FGR)", "CPD配准", "PCA配准",
        "点对点ICP", "点对面ICP", "广义ICP(GICP)", "彩色ICP",
    ]

    def __init__(self):
        self.algorithm: str = "点对点ICP"
        # 粗配准参数
        self.voxel_size: float = 0.05
        self.distance_threshold: float = 0.05
        self.ransac_n: int = 3
        self.max_iterations: int = 2000
        self.division_factor: float = 1.4
        self.decrease_mu: bool = True
        # 精配准参数
        self.relative_fitness: float = 1e-6
        self.relative_rmse: float = 1e-6
        # CPD 参数
        self.cpd_tol: float = 1e-4
        # 评估指标（运行后填充）
        self._last_rmse: float = 0.0
        self._last_fitness: float = 0.0
        self._last_transformation: Optional[np.ndarray] = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云", multi_connect=True)
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        import open3d as o3d

        pcd_list = inputs.get("点云")
        if not pcd_list or len(pcd_list) < 2:
            raise ValueError("至少需要两个点云输入用于配准")

        target = pcd_list[0]  # 第一个作为目标
        results = [target]

        params = {
            "voxel_size": self.voxel_size,
            "distance_threshold": self.distance_threshold,
            "ransac_n": self.ransac_n,
            "max_iterations": self.max_iterations,
            "division_factor": self.division_factor,
            "decrease_mu": self.decrease_mu,
            "relative_fitness": self.relative_fitness,
            "relative_rmse": self.relative_rmse,
            "cpd_tol": self.cpd_tol,
        }

        coarse_algos = {"FPFH+RANSAC配准", "快速全局配准(FGR)", "CPD配准", "PCA配准"}
        fine_algos = {"点对点ICP", "点对面ICP", "广义ICP(GICP)", "彩色ICP"}

        total_rmse = 0.0
        total_fitness = 0.0
        count = 0

        for i, source in enumerate(pcd_list[1:]):
            if self.algorithm in coarse_algos:
                aligned, result = _do_coarse_registration(source, target, self.algorithm, params)
            elif self.algorithm in fine_algos:
                aligned, result = _do_fine_registration(source, target, self.algorithm, params)

            if hasattr(result, 'inlier_rmse'):
                total_rmse += result.inlier_rmse
            if hasattr(result, 'fitness'):
                total_fitness += result.fitness
            count += 1
            results.append(aligned)

        if count > 0:
            self._last_rmse = total_rmse / count
            self._last_fitness = total_fitness / count

        combined = o3d.geometry.PointCloud()
        combined_points = []
        combined_colors = []
        np.random.seed(42)
        for idx, r in enumerate(results):
            pts = np.asarray(r.points)
            color = np.array([[0.7, 0.7, 0.7]] if idx == 0
                             else [np.random.rand(3)])
            combined_points.append(pts)
            combined_colors.append(np.tile(color, (len(pts), 1)))
        combined.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))

        return {"点云": combined}
