"""图像平滑算子：高斯模糊 / 均值滤波 / 中值滤波 / 双边滤波 / 引导滤波。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


def _guided_filter(img, radius: int, eps: float):
    """引导滤波（自引导）。"""
    I = img.astype(np.float32) / 255.0
    r = radius
    ksize = (r, r)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize, normalize=True)
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize, normalize=True)
    var_I = mean_II - mean_I * mean_I

    mean_p = cv2.boxFilter(I, cv2.CV_32F, ksize, normalize=True)
    mean_Ip = cv2.boxFilter(I * I, cv2.CV_32F, ksize, normalize=True)
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True)

    q = mean_a * I + mean_b
    q = np.clip(q * 255, 0, 255).astype(np.uint8)
    return q


@register_node
class SmoothingNode(Node):
    display_name = "图像平滑"
    category = "图像处理"
    algorithms = ["高斯模糊", "均值滤波", "中值滤波", "双边滤波", "引导滤波"]

    def __init__(self):
        self.algorithm: str = "高斯模糊"
        self.kernel_size: int = 5
        self.sigma_color: float = 75.0
        self.sigma_space: float = 75.0
        self.epsilon: float = 0.01
        self.radius: int = 4
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

        if self.algorithm == "高斯模糊":
            result = cv2.GaussianBlur(img, (k, k), 0)
        elif self.algorithm == "均值滤波":
            result = cv2.blur(img, (k, k))
        elif self.algorithm == "中值滤波":
            result = cv2.medianBlur(img, k)
        elif self.algorithm == "双边滤波":
            if len(img.shape) == 3:
                result = cv2.bilateralFilter(img, k, self.sigma_color, self.sigma_space)
            else:
                result = cv2.bilateralFilter(img, k, self.sigma_color, self.sigma_space)
        elif self.algorithm == "引导滤波":
            if len(img.shape) == 3:
                result = np.zeros_like(img)
                for c in range(3):
                    result[:, :, c] = _guided_filter(img[:, :, c], self.radius, self.epsilon)
            else:
                result = _guided_filter(img, self.radius, self.epsilon)
        else:
            result = img

        return {"图像": result}
