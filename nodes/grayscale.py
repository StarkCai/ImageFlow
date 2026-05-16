"""灰度化算子：将彩色图像转换为灰度图，支持多种转换方法。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


@register_node
class GrayscaleNode(Node):
    display_name = "灰度化"
    category = "图像处理"

    def __init__(self):
        self.method: str = "luminosity"
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if len(img.shape) == 2:
            gray = img
        elif self.method == "average":
            gray = np.mean(img, axis=2).astype(np.uint8)
        elif self.method == "lightness":
            gray = ((np.max(img, axis=2) + np.min(img, axis=2)) // 2).astype(np.uint8)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return {"图像": gray}
