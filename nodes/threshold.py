"""阈值二值化算子：Binary阈值 / Otsu阈值 / 自适应阈值。"""

import cv2
from node_base import Node
from node_registry import register_node


@register_node
class ThresholdNode(Node):
    display_name = "阈值二值化"
    category = "图像处理"
    algorithms = ["Binary阈值", "Otsu阈值", "自适应阈值"]

    def __init__(self):
        self.algorithm: str = "Binary阈值"
        self.threshold_value: int = 127
        self.max_value: int = 255
        self.block_size: int = 11
        self.C: int = 2
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

        if self.algorithm == "Binary阈值":
            _, result = cv2.threshold(
                gray, self.threshold_value, self.max_value, cv2.THRESH_BINARY
            )
        elif self.algorithm == "Otsu阈值":
            _, result = cv2.threshold(
                gray, 0, self.max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif self.algorithm == "自适应阈值":
            block = self.block_size
            if block % 2 == 0:
                block += 1
            result = cv2.adaptiveThreshold(
                gray, self.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, self.C,
            )
        else:
            result = gray

        return {"图像": result}
