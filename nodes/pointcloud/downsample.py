"""点云下采样算子：体素下采样 / 均匀下采样 / 随机下采样 / 最远点采样。"""

from __future__ import annotations

from node_base import Node
from node_registry import register_node


@register_node
class PointCloudDownsampleNode(Node):
    display_name = "点云下采样"
    category = "点云处理"
    algorithms = ["体素下采样", "均匀下采样", "随机下采样", "最远点采样"]

    def __init__(self):
        self.algorithm: str = "体素下采样"
        # 体素下采样
        self.voxel_size: float = 0.05
        # 均匀下采样
        self.every_k_points: int = 5
        # 随机下采样
        self.sampling_ratio: float = 0.5
        # 最远点采样
        self.num_samples: int = 1024
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")
        self.add_output("点云", data_type="点云")

    def process(self, **inputs):
        import open3d as o3d

        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        if self.algorithm == "体素下采样":
            return {"点云": pcd.voxel_down_sample(self.voxel_size)}
        elif self.algorithm == "均匀下采样":
            return {"点云": pcd.uniform_down_sample(self.every_k_points)}
        elif self.algorithm == "随机下采样":
            return {"点云": pcd.random_down_sample(self.sampling_ratio)}
        elif self.algorithm == "最远点采样":
            result = pcd.farthest_point_down_sample(self.num_samples)
            return {"点云": result}
        return {"点云": pcd}
