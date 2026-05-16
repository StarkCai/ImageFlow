"""图像缩放算子。"""

import cv2
from node_base import Node
from node_registry import register_node


@register_node
class ResizeNode(Node):
    display_name = "图像缩放"
    category = "图像处理"

    INTERP_MAP = {
        "最近邻": cv2.INTER_NEAREST,
        "双线性": cv2.INTER_LINEAR,
        "双三次": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(self):
        self.width: int = 256
        self.height: int = 256
        self.keep_aspect: bool = True
        self.interpolation: str = "双线性"
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if self.keep_aspect:
            h, w = img.shape[:2]
            scale = min(self.width / w, self.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = self.width, self.height

        interp = self.INTERP_MAP.get(self.interpolation, cv2.INTER_LINEAR)
        result = cv2.resize(img, (new_w, new_h), interpolation=interp)
        return {"图像": result}
