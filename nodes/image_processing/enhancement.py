"""图像增强算子：直方图均衡化 / 对比度拉伸 / 伽马校正 / 对数变换 / 锐化滤波。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


@register_node
class EnhancementNode(Node):
    display_name = "图像增强"
    category = "图像处理"
    algorithms = ["直方图均衡化", "对比度拉伸", "伽马校正", "对数变换", "锐化滤波"]

    def __init__(self):
        self.algorithm: str = "直方图均衡化"
        # 对比度拉伸
        self.clip_low_pct: float = 2.0
        self.clip_high_pct: float = 2.0
        # 伽马校正
        self.gamma: float = 1.0
        # 对数变换
        self.log_c: float = 1.0
        # 锐化滤波
        self.sharpen_amount: float = 1.0
        self.sharpen_radius: int = 1
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if self.algorithm == "直方图均衡化":
            return self._hist_eq(img)
        elif self.algorithm == "对比度拉伸":
            return self._contrast_stretch(img)
        elif self.algorithm == "伽马校正":
            return self._gamma_correct(img)
        elif self.algorithm == "对数变换":
            return self._log_transform(img)
        elif self.algorithm == "锐化滤波":
            return self._sharpen(img)
        else:
            return {"图像": img}

    def _hist_eq(self, img):
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            result = cv2.equalizeHist(img)
        return {"图像": result}

    def _contrast_stretch(self, img):
        low = np.percentile(img, self.clip_low_pct)
        high = np.percentile(img, 100 - self.clip_high_pct)
        result = np.clip((img.astype(np.float32) - low) * 255.0 / max(high - low, 1), 0, 255)
        return {"图像": result.astype(np.uint8)}

    def _gamma_correct(self, img):
        table = (np.arange(256) / 255.0) ** (1.0 / max(self.gamma, 0.01))
        table = (table * 255).astype(np.uint8)
        result = cv2.LUT(img, table)
        return {"图像": result}

    def _log_transform(self, img):
        c = self.log_c
        norm = img.astype(np.float32) / 255.0
        result = c * np.log1p(norm)
        result = np.clip(result / np.max(result) * 255, 0, 255) if np.max(result) > 0 else result
        return {"图像": result.astype(np.uint8)}

    def _sharpen(self, img):
        r = self.sharpen_radius
        ksize = 2 * r + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        result = cv2.addWeighted(img, 1.0 + self.sharpen_amount, blurred,
                                 -self.sharpen_amount, 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return {"图像": result}
