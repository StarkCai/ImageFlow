"""Canny边缘检测算子。"""

import cv2
from node_base import Node
from node_registry import register_node


@register_node
class EdgeDetectNode(Node):
    display_name = "边缘检测"
    category = "图像处理"

    def __init__(self):
        self.threshold1: int = 50
        self.threshold2: int = 150
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        return {"图像": edges}
