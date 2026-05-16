"""边缘检测算子：Sobel / Canny / Laplacian / Roberts。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


@register_node
class EdgeDetectNode(Node):
    display_name = "边缘检测"
    category = "图像处理"
    algorithms = ["Sobel", "Canny", "Laplacian", "Roberts"]

    def __init__(self):
        self.algorithm: str = "Canny"
        # 通用
        self.kernel_size: int = 3
        # Canny
        self.threshold_low: int = 50
        self.threshold_high: int = 150
        self.sigma: float = 1.0
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

        if self.algorithm == "Sobel":
            return self._sobel(gray)
        elif self.algorithm == "Canny":
            return self._canny(gray)
        elif self.algorithm == "Laplacian":
            return self._laplacian(gray)
        elif self.algorithm == "Roberts":
            return self._roberts(gray)
        else:
            return {"图像": gray}

    def _sobel(self, gray):
        k = self.kernel_size
        if k % 2 == 0:
            k += 1
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        result = cv2.magnitude(gx, gy)
        return {"图像": np.uint8(np.clip(result, 0, 255))}

    def _canny(self, gray):
        k = self.kernel_size
        if k % 2 == 0:
            k += 1
        if self.sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.sigma)
        edges = cv2.Canny(gray, self.threshold_low, self.threshold_high, apertureSize=k)
        return {"图像": edges}

    def _laplacian(self, gray):
        k = self.kernel_size
        if k % 2 == 0:
            k += 1
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        return {"图像": np.uint8(np.absolute(lap))}

    def _roberts(self, gray):
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, kx)
        gy = cv2.filter2D(gray.astype(np.float32), cv2.CV_64F, ky)
        result = cv2.magnitude(gx, gy)
        return {"图像": np.uint8(np.clip(result, 0, 255))}
