"""图像读取算子：从文件路径读取图像，支持中文路径。"""

import os
from typing import Optional

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node


def _imread_unicode(filepath: str) -> Optional[np.ndarray]:
    """支持中文路径的图像读取。"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@register_node
class ImageInputNode(Node):
    display_name = "图像读取"
    category = "输入"

    def __init__(self):
        self._file_path: str = ""
        super().__init__()

    def _setup_ports(self):
        self.add_output("图像")

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, path: str):
        self._file_path = path

    def process(self, **inputs):
        if not self._file_path:
            raise ValueError("未设置图像文件路径")

        img = _imread_unicode(self._file_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self._file_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {"图像": img_rgb}
