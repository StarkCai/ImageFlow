"""形态学操作算子：腐蚀 / 膨胀 / 开运算 / 闭运算 / 形态学梯度。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


KERNEL_SHAPE = {
    "矩形": cv2.MORPH_RECT,
    "椭圆": cv2.MORPH_ELLIPSE,
    "十字": cv2.MORPH_CROSS,
}


@register_node
class MorphologyNode(Node):
    display_name = "形态学操作"
    category = "图像处理"
    algorithms = ["腐蚀", "膨胀", "开运算", "闭运算", "形态学梯度"]

    def __init__(self):
        self.algorithm: str = "腐蚀"
        self.kernel_size: int = 3
        self.kernel_shape: str = "矩形"
        self.iterations: int = 1
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        k = self.kernel_size
        if k % 2 == 0:
            k += 1
        shape = KERNEL_SHAPE.get(self.kernel_shape, cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (k, k))

        op_map = {
            "腐蚀": cv2.MORPH_ERODE,
            "膨胀": cv2.MORPH_DILATE,
            "开运算": cv2.MORPH_OPEN,
            "闭运算": cv2.MORPH_CLOSE,
            "形态学梯度": cv2.MORPH_GRADIENT,
        }
        op = op_map.get(self.algorithm, cv2.MORPH_ERODE)
        result = cv2.morphologyEx(img, op, kernel, iterations=self.iterations)
        return {"图像": result}
