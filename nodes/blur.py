"""高斯模糊算子：对图像进行高斯平滑处理。"""

import cv2
from node_base import Node
from node_registry import register_node


@register_node
class BlurNode(Node):
    display_name = "高斯模糊"
    category = "图像处理"

    def __init__(self):
        self.kernel_size: int = 5
        self.sigma: float = 1.0
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        # 确保核大小为奇数
        ksize = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        result = cv2.GaussianBlur(img, (ksize, ksize), self.sigma)
        return {"图像": result}
