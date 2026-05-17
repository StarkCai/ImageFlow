"""图像分割算子：区域生长 / 分水岭 / K-means / GrabCut / 阈值分割。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node


def _region_growing(gray, seed_x, seed_y, threshold):
    """简单的区域生长算法。"""
    h, w = gray.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    result = np.zeros((h, w), dtype=np.uint8)

    seed_val = gray[seed_y, seed_x]
    stack = [(seed_x, seed_y)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[y, x]:
            continue
        if abs(int(gray[y, x]) - int(seed_val)) > threshold:
            continue
        visited[y, x] = 1
        result[y, x] = 255
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            stack.append((x + dx, y + dy))

    return result


@register_node
class SegmentationNode(Node):
    display_name = "图像分割"
    category = "图像处理"
    algorithms = ["区域生长", "分水岭算法", "K-means聚类", "GrabCut", "阈值分割"]

    def __init__(self):
        self.algorithm: str = "阈值分割"
        # 区域生长
        self.seed_x: int = 100
        self.seed_y: int = 100
        self.grow_threshold: int = 5
        # K-means
        self.k: int = 3
        self.kmeans_attempts: int = 5
        self.kmeans_max_iter: int = 10
        # GrabCut
        self.grabcut_iters: int = 5
        # 阈值分割
        self.seg_threshold: int = 127
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

        if self.algorithm == "区域生长":
            result = _region_growing(gray, self.seed_x, self.seed_y, self.grow_threshold)
        elif self.algorithm == "分水岭算法":
            result = self._watershed(img)
        elif self.algorithm == "K-means聚类":
            result = self._kmeans(img)
        elif self.algorithm == "GrabCut":
            result = self._grabcut(img)
        elif self.algorithm == "阈值分割":
            _, result = cv2.threshold(gray, self.seg_threshold, 255, cv2.THRESH_BINARY)
        else:
            result = gray

        return {"图像": result}

    def _watershed(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 噪声去除
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 确定前景区域
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers.copy())
        boundary = np.uint8(markers == -1) * 255
        return boundary

    def _kmeans(self, img):
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    self.kmeans_max_iter, 1.0)
        _, labels, centers = cv2.kmeans(
            data, self.k, None, criteria, self.kmeans_attempts, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape(img.shape)
        return result

    def _grabcut(self, img):
        h, w = img.shape[:2]
        rect = (max(1, w // 10), max(1, h // 10),
                max(1, w - 2 * w // 10), max(1, h - 2 * h // 10))
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, self.grabcut_iters,
                    cv2.GC_INIT_WITH_RECT)
        result = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)) * 255
        return result
