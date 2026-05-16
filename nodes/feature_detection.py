"""特征检测算子：Harris角点 / FAST / SIFT / SURF / HOG。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


def _draw_keypoints(img, kps, color=(0, 255, 0)):
    """在图像上绘制关键点。"""
    if isinstance(kps, np.ndarray):
        result = img.copy()
        for pt in kps:
            x, y = int(pt[0][0]), int(pt[0][1])
            cv2.circle(result, (x, y), 4, color, 1)
        return result
    return cv2.drawKeypoints(img, kps, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def _hog_visualize(img, cell_size=8, block_size=2, nbins=9):
    """计算 HOG 特征并可视化。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    h, w = gray.shape
    result = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    stride = cell_size
    for y in range(0, h - cell_size + 1, stride):
        for x in range(0, w - cell_size + 1, stride):
            cell_mag = mag[y:y + cell_size, x:x + cell_size]
            cell_angle = angle[y:y + cell_size, x:x + cell_size]
            hist = np.zeros(nbins)
            bin_width = 180.0 / nbins
            for cy in range(cell_size):
                for cx in range(cell_size):
                    bidx = min(int(cell_angle[cy, cx] / bin_width), nbins - 1)
                    hist[bidx] += cell_mag[cy, cx]

            cx_p = x + cell_size // 2
            cy_p = y + cell_size // 2
            for b in range(nbins):
                if hist[b] > 0:
                    theta = np.deg2rad(b * bin_width + bin_width / 2)
                    length = min(hist[b] * 0.05, cell_size * 0.6)
                    dx = length * np.cos(theta)
                    dy = length * np.sin(theta)
                    cv2.line(result,
                             (int(cx_p - dx), int(cy_p - dy)),
                             (int(cx_p + dx), int(cy_p + dy)),
                             (0, 200, 255), 1)

    return result


@register_node
class FeatureDetectionNode(Node):
    display_name = "特征检测"
    category = "图像处理"
    algorithms = ["Harris角点检测", "FAST角点检测", "SIFT特征", "SURF特征", "HOG特征"]

    def __init__(self):
        self.algorithm: str = "Harris角点检测"
        # Harris
        self.harris_block_size: int = 2
        self.harris_ksize: int = 3
        self.harris_k: float = 0.04
        self.harris_threshold: float = 0.01
        # FAST
        self.fast_threshold: int = 50
        self.fast_nonmax: bool = True
        # SIFT
        self.sift_nfeatures: int = 0
        self.sift_n_octave_layers: int = 3
        self.sift_contrast_threshold: float = 0.04
        self.sift_edge_threshold: float = 10.0
        self.sift_sigma: float = 1.6
        # SURF
        self.surf_hessian: int = 400
        self.surf_n_octaves: int = 4
        self.surf_n_octave_layers: int = 3
        self.surf_extended: bool = True
        self.surf_upright: bool = False
        # HOG
        self.hog_cell_size: int = 8
        self.hog_block_size: int = 2
        self.hog_nbins: int = 9
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        if self.algorithm == "Harris角点检测":
            result = self._harris(img)
        elif self.algorithm == "FAST角点检测":
            result = self._fast(img)
        elif self.algorithm == "SIFT特征":
            result = self._sift(img)
        elif self.algorithm == "SURF特征":
            result = self._surf(img)
        elif self.algorithm == "HOG特征":
            result = self._hog(img)
        else:
            result = img

        return {"图像": result}

    def _harris(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        gray_f = np.float32(gray)
        ksize = self.harris_ksize
        if ksize % 2 == 0:
            ksize += 1
        dst = cv2.cornerHarris(gray_f, self.harris_block_size, ksize, self.harris_k)
        result = img.copy()
        result[dst > self.harris_threshold * dst.max()] = [255, 0, 0]
        return result

    def _fast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        fast = cv2.FastFeatureDetector_create(
            threshold=self.fast_threshold,
            nonmaxSuppression=self.fast_nonmax,
        )
        kps = fast.detect(gray, None)
        return _draw_keypoints(img, kps)

    def _sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        sift = cv2.SIFT_create(
            nfeatures=self.sift_nfeatures or None,
            nOctaveLayers=self.sift_n_octave_layers,
            contrastThreshold=self.sift_contrast_threshold,
            edgeThreshold=self.sift_edge_threshold,
            sigma=self.sift_sigma,
        )
        kps = sift.detect(gray, None)
        return _draw_keypoints(img, kps)

    def _surf(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=self.surf_hessian,
            nOctaves=self.surf_n_octaves,
            nOctaveLayers=self.surf_n_octave_layers,
            extended=self.surf_extended,
            upright=self.surf_upright,
        )
        kps = surf.detect(gray, None)
        return _draw_keypoints(img, kps)

    def _hog(self, img):
        return _hog_visualize(img, self.hog_cell_size, self.hog_block_size, self.hog_nbins)
