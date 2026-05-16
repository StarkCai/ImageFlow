"""图像绘制（叠加区域）算子：将区域数据绘制到图像上，支持轮廓/实心/半透明三种风格。"""

from typing import List

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node

# 颜色表（BGR 格式，用于 OpenCV 绘制）
REGION_COLORS = [
    (255, 80, 80),     # 红色
    (80, 160, 255),    # 蓝色
    (80, 255, 130),    # 绿色
    (255, 200, 50),    # 金色
    (200, 80, 255),    # 紫色
    (50, 200, 255),    # 青色
    (255, 130, 80),    # 橙色
    (180, 180, 180),   # 灰色
]


def _flatten_regions(regions_raw) -> list:
    """扁平化区域数据：处理单/多连线场景。

    regions_raw 可能是：
      - list[dict]：单个区域列表（单连线）
      - list[list[dict]]：多个区域列表（多连线）
    """
    if not regions_raw:
        return []
    flat = []
    for item in regions_raw:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def _draw_region_outline(img_bgr: np.ndarray, region: dict, color: tuple, thickness: int = 2):
    """轮廓绘制。"""
    rtype = region["type"]
    coords = region["coordinates"]

    if rtype == "矩形":
        cv2.rectangle(img_bgr, (coords["x1"], coords["y1"]),
                      (coords["x2"], coords["y2"]), color, thickness)
    elif rtype == "圆形":
        cv2.circle(img_bgr, (coords["cx"], coords["cy"]),
                   coords["radius"], color, thickness)
    elif rtype == "多边形":
        pts = np.array(coords["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, color, thickness)


def _draw_region_fill(img_bgr: np.ndarray, region: dict, color: tuple):
    """实心填充。"""
    rtype = region["type"]
    coords = region["coordinates"]

    if rtype == "矩形":
        cv2.rectangle(img_bgr, (coords["x1"], coords["y1"]),
                      (coords["x2"], coords["y2"]), color, -1)
    elif rtype == "圆形":
        cv2.circle(img_bgr, (coords["cx"], coords["cy"]),
                   coords["radius"], color, -1)
    elif rtype == "多边形":
        pts = np.array(coords["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img_bgr, [pts], color)


@register_node
class OverlayNode(Node):
    display_name = "图像绘制"
    category = "图像叠加"
    algorithms = ["轮廓绘制", "实心填充", "半透明填充"]

    def __init__(self):
        self.algorithm: str = "轮廓绘制"
        self.alpha: float = 0.4
        self.thickness: int = 2
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_input("区域", multi_connect=True)
        self.add_output("图像")

    def process(self, **inputs):
        img_rgb = inputs.get("图像")
        if img_rgb is None:
            raise ValueError("未接收到图像数据")

        regions_raw = inputs.get("区域")
        regions = _flatten_regions(regions_raw)
        if not regions:
            raise ValueError("未接收到区域数据")

        # RGB → BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        result = img_bgr.copy()

        if self.algorithm == "半透明填充":
            overlay = np.zeros_like(img_bgr)

        for i, region in enumerate(regions):
            color = REGION_COLORS[i % len(REGION_COLORS)]

            if self.algorithm == "轮廓绘制":
                _draw_region_outline(result, region, color, self.thickness)
            elif self.algorithm == "实心填充":
                _draw_region_fill(result, region, color)
            elif self.algorithm == "半透明填充":
                _draw_region_fill(overlay, region, color)

        if self.algorithm == "半透明填充":
            result = cv2.addWeighted(result, 1.0 - self.alpha, overlay,
                                     self.alpha, 0)

        # BGR → RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return {"图像": result_rgb}
