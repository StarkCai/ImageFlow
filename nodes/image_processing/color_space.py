"""颜色空间转换算子：支持主流颜色空间之间的转换。"""

import cv2
from node_base import Node
from node_registry import register_node

COLOR_CODES = {
    "RGB → HSV": cv2.COLOR_RGB2HSV,
    "RGB → LAB": cv2.COLOR_RGB2LAB,
    "RGB → YCrCb": cv2.COLOR_RGB2YCrCb,
    "RGB → YUV": cv2.COLOR_RGB2YUV,
    "RGB → HLS": cv2.COLOR_RGB2HLS,
    "RGB → LUV": cv2.COLOR_RGB2LUV,
    "RGB → Gray": cv2.COLOR_RGB2GRAY,
    "RGB → XYZ": cv2.COLOR_RGB2XYZ,
    "HSV → RGB": cv2.COLOR_HSV2RGB,
    "LAB → RGB": cv2.COLOR_LAB2RGB,
    "BGR → RGB": cv2.COLOR_BGR2RGB,
    "RGB → BGR": cv2.COLOR_RGB2BGR,
}


@register_node
class ColorSpaceNode(Node):
    display_name = "颜色空间转换"
    category = "图像处理"
    algorithms = list(COLOR_CODES.keys())

    def __init__(self):
        self.algorithm: str = "RGB → HSV"
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        code = COLOR_CODES.get(self.algorithm)
        if code is None:
            return {"图像": img}

        # 灰度转换不需要 3 通道输入
        if self.algorithm == "RGB → Gray":
            if len(img.shape) == 3:
                result = cv2.cvtColor(img, code)
            else:
                result = img
            return {"图像": result}

        # 其他转换需要 3 通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        result = cv2.cvtColor(img, code)
        return {"图像": result}
