"""二值化阈值算子。"""

import cv2
from node_base import Node
from node_registry import register_node


@register_node
class ThresholdNode(Node):
    display_name = "阈值二值化"
    category = "图像处理"

    def __init__(self):
        self.threshold_value: int = 127
        self.max_value: int = 255
        self.method: str = "binary"  # binary, binary_inv, trunc, tozero, otsu
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def _get_method_code(self) -> int:
        mapping = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        }
        return mapping.get(self.method, cv2.THRESH_BINARY)

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        thresh_val = self.threshold_value
        if self.method == "otsu":
            thresh_val = 0

        _, result = cv2.threshold(gray, thresh_val, self.max_value, self._get_method_code())
        return {"图像": result}
