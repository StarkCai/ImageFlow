"""几何变换算子：仿射变换 / 透视变换 / 图像旋转 / 图像缩放。"""

import cv2
import numpy as np
from node_base import Node
from node_registry import register_node

INTERP_MAP = {
    "最近邻": cv2.INTER_NEAREST,
    "双线性": cv2.INTER_LINEAR,
    "双三次": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
}


@register_node
class GeometryTransformNode(Node):
    display_name = "几何变换"
    category = "图像处理"
    algorithms = ["仿射变换", "透视变换", "图像旋转", "图像缩放"]

    def __init__(self):
        self.algorithm: str = "图像缩放"

        # 仿射变换
        self.affine_angle: float = 0.0
        self.affine_scale_x: float = 1.0
        self.affine_scale_y: float = 1.0
        self.affine_tx: int = 0
        self.affine_ty: int = 0

        # 透视变换（相对偏移，百分比 0–100）
        self.persp_tl_x: int = 0
        self.persp_tl_y: int = 0
        self.persp_tr_x: int = 0
        self.persp_tr_y: int = 0
        self.persp_bl_x: int = 0
        self.persp_bl_y: int = 0
        self.persp_br_x: int = 0
        self.persp_br_y: int = 0

        # 图像旋转
        self.rotate_angle: float = 45.0
        self.rotate_scale: float = 1.0

        # 图像缩放
        self.scale_width: int = 256
        self.scale_height: int = 256
        self.keep_aspect: bool = True
        self.interpolation: str = "双线性"

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")
        self.add_output("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        h, w = img.shape[:2]

        if self.algorithm == "仿射变换":
            return self._affine(img, w, h)
        elif self.algorithm == "透视变换":
            return self._perspective(img, w, h)
        elif self.algorithm == "图像旋转":
            return self._rotate(img, w, h)
        elif self.algorithm == "图像缩放":
            return self._scale(img, w, h)
        else:
            return {"图像": img}

    def _affine(self, img, w, h):
        angle = np.deg2rad(self.affine_angle)
        M = np.array([
            [self.affine_scale_x * np.cos(angle),
             -self.affine_scale_y * np.sin(angle), self.affine_tx],
            [self.affine_scale_x * np.sin(angle),
             self.affine_scale_y * np.cos(angle), self.affine_ty],
        ], dtype=np.float32)
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        return {"图像": result}

    def _perspective(self, img, w, h):
        src = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        ], dtype=np.float32)
        dst = np.array([
            [self.persp_tl_x, self.persp_tl_y],
            [w - 1 + self.persp_tr_x, self.persp_tr_y],
            [w - 1 + self.persp_br_x, h - 1 + self.persp_br_y],
            [self.persp_bl_x, h - 1 + self.persp_bl_y],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        return {"图像": result}

    def _rotate(self, img, w, h):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), self.rotate_angle, self.rotate_scale)
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        return {"图像": result}

    def _scale(self, img, w, h):
        if self.keep_aspect:
            scale = min(self.scale_width / w, self.scale_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = self.scale_width, self.scale_height

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        interp = INTERP_MAP.get(self.interpolation, cv2.INTER_LINEAR)
        result = cv2.resize(img, (new_w, new_h), interpolation=interp)
        return {"图像": result}
