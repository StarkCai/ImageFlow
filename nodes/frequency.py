"""频域处理算子：傅里叶变换 / 高通滤波 / 低通滤波 / 带通滤波 / 同态滤波。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


def _fft(gray):
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def _ifft(dft_shift):
    dft = np.fft.ifftshift(dft_shift)
    result = cv2.idft(dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return np.clip(result, 0, 255).astype(np.uint8)


def _gaussian_mask(rows, cols, center_y, center_x, cutoff, high_pass=False):
    ys = np.arange(rows).reshape(-1, 1).astype(np.float32)
    xs = np.arange(cols).reshape(1, -1).astype(np.float32)
    D2 = (ys - center_y) ** 2 + (xs - center_x) ** 2
    D = np.sqrt(D2)
    mask = 1.0 - np.exp(-(D ** 2) / (2 * cutoff ** 2)) if high_pass \
        else np.exp(-(D ** 2) / (2 * cutoff ** 2))
    return np.dstack([mask, mask])


def _bandpass_mask(rows, cols, cy, cx, low_cut, high_cut):
    ys = np.arange(rows).reshape(-1, 1).astype(np.float32)
    xs = np.arange(cols).reshape(1, -1).astype(np.float32)
    D = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    low = np.exp(-(D ** 2) / (2 * low_cut ** 2))
    high = 1.0 - np.exp(-(D ** 2) / (2 * high_cut ** 2))
    mask = low * high
    return np.dstack([mask, mask])


@register_node
class FrequencyNode(Node):
    display_name = "频域处理"
    category = "图像处理"
    algorithms = ["傅里叶变换", "高通滤波", "低通滤波", "带通滤波", "同态滤波"]

    def __init__(self):
        self.algorithm: str = "傅里叶变换"
        # 高/低/带通
        self.cutoff: int = 30
        self.low_cutoff: int = 10
        self.high_cutoff: int = 50
        # 同态滤波
        self.homo_low_gamma: float = 0.5
        self.homo_high_gamma: float = 2.0
        self.homo_cutoff: int = 30
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        rows, cols = gray.shape
        cy, cx = rows // 2, cols // 2

        if self.algorithm == "傅里叶变换":
            dft = _fft(gray)
            mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
            result = np.uint8(np.clip(np.log1p(mag) * 20, 0, 255))
        elif self.algorithm == "高通滤波":
            dft = _fft(gray)
            mask = _gaussian_mask(rows, cols, cy, cx, self.cutoff, high_pass=True)
            result = _ifft(dft * mask)
        elif self.algorithm == "低通滤波":
            dft = _fft(gray)
            mask = _gaussian_mask(rows, cols, cy, cx, self.cutoff, high_pass=False)
            result = _ifft(dft * mask)
        elif self.algorithm == "带通滤波":
            dft = _fft(gray)
            mask = _bandpass_mask(rows, cols, cy, cx, self.low_cutoff, self.high_cutoff)
            result = _ifft(dft * mask)
        elif self.algorithm == "同态滤波":
            result = self._homomorphic(gray, rows, cols, cy, cx)
        else:
            result = gray

        return {"图像": result}

    def _homomorphic(self, gray, rows, cols, cy, cx):
        img_log = np.log1p(gray.astype(np.float32))
        dft = _fft(img_log)

        # 同态滤波器：低频衰减 + 高频增强
        ys = np.arange(rows).reshape(-1, 1).astype(np.float32)
        xs = np.arange(cols).reshape(1, -1).astype(np.float32)
        D2 = (ys - cy) ** 2 + (xs - cx) ** 2
        D = np.sqrt(D2)
        gh = self.homo_high_gamma
        gl = self.homo_low_gamma
        c = self.homo_cutoff
        H = (gh - gl) * (1.0 - np.exp(-c * D2 / (D2.max() + 1e-5))) + gl
        mask = np.dstack([H, H])

        result = _ifft(dft * mask)
        result = np.expm1(result.astype(np.float32))
        return np.clip(result, 0, 255).astype(np.uint8)
