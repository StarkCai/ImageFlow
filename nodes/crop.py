"""图像裁剪算子：根据区域裁剪图像，支持直接裁剪和区域掩膜两种方式。"""

from typing import List

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node


def _region_bounding_rect(region: dict) -> tuple:
    """计算单个区域的包围矩形 (x1, y1, x2, y2)。"""
    coords = region["coordinates"]
    rtype = region["type"]

    if rtype == "矩形":
        return (coords["x1"], coords["y1"], coords["x2"], coords["y2"])
    elif rtype == "圆形":
        cx, cy, r = coords["cx"], coords["cy"], coords["radius"]
        return (cx - r, cy - r, cx + r, cy + r)
    elif rtype == "多边形":
        pts = coords["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return (0, 0, 0, 0)


def _union_bounding_rect(regions: list) -> tuple:
    """计算所有区域的合并包围矩形，返回 (x, y, w, h)。"""
    if not regions:
        return (0, 0, 1, 1)
    first = _region_bounding_rect(regions[0])
    x1, y1, x2, y2 = first
    for r in regions[1:]:
        rx1, ry1, rx2, ry2 = _region_bounding_rect(r)
        x1 = min(x1, rx1)
        y1 = min(y1, ry1)
        x2 = max(x2, rx2)
        y2 = max(y2, ry2)
    return (x1, y1, x2 - x1, y2 - y1)


def _build_mask(regions: list, h: int, w: int) -> np.ndarray:
    """为所有区域构建二值掩膜。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        coords = region["coordinates"]
        rtype = region["type"]
        if rtype == "矩形":
            cv2.rectangle(mask, (coords["x1"], coords["y1"]),
                          (coords["x2"], coords["y2"]), 255, -1)
        elif rtype == "圆形":
            cv2.circle(mask, (coords["cx"], coords["cy"]),
                       coords["radius"], 255, -1)
        elif rtype == "多边形":
            pts = np.array(coords["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
    return mask


@register_node
class CropNode(Node):
    display_name = "图像裁剪"
    category = "图像叠加"
    algorithms = ["区域裁剪", "区域掩膜"]

    def __init__(self):
        self.algorithm: str = "区域裁剪"
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_input("区域")
        self.add_output("图像")

    def process(self, **inputs):
        img_rgb = inputs.get("图像")
        if img_rgb is None:
            raise ValueError("未接收到图像数据")

        regions_raw = inputs.get("区域")
        if not regions_raw:
            raise ValueError("未接收到区域数据")

        # 处理单区域 dict 或区域列表
        if isinstance(regions_raw, dict):
            regions = [regions_raw]
        elif isinstance(regions_raw, list) and regions_raw and isinstance(regions_raw[0], dict):
            regions = regions_raw
        elif isinstance(regions_raw, list) and regions_raw and isinstance(regions_raw[0], list):
            # 列表的列表（来自多连线输入），取第一个
            regions = regions_raw[0] if isinstance(regions_raw[0], list) else [regions_raw[0]]
        else:
            regions = []

        if not regions:
            raise ValueError("区域数据为空")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if self.algorithm == "区域裁剪":
            result = self._crop(img_bgr, regions)
        elif self.algorithm == "区域掩膜":
            result = self._mask_crop(img_bgr, regions)
        else:
            result = img_bgr

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return {"图像": result_rgb}

    def _crop(self, img_bgr: np.ndarray, regions: list) -> np.ndarray:
        """直接裁剪：用包围矩形裁切图像。"""
        h, w = img_bgr.shape[:2]
        x, y, rw, rh = _union_bounding_rect(regions)
        x = max(0, x)
        y = max(0, y)
        rw = min(rw, w - x)
        rh = min(rh, h - y)
        if rw < 1 or rh < 1:
            return img_bgr
        return img_bgr[y:y + rh, x:x + rw].copy()

    def _mask_crop(self, img_bgr: np.ndarray, regions: list) -> np.ndarray:
        """掩膜裁剪：区域外置为黑色，区域内保留原图。"""
        h, w = img_bgr.shape[:2]
        mask = _build_mask(regions, h, w)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        return result
