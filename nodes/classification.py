"""图像分类算子：基于 ONNX Runtime 通用推理框架，输入图像输出叠加分类标签的图像。"""

import os
from typing import List, Tuple

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node


@register_node
class ClassificationNode(Node):
    display_name = "目标分类"
    category = "深度学习"
    algorithms = ["目标分类"]

    def __init__(self):
        self.model_path: str = ""

        # 模型输入分辨率
        self.input_width: int = 224
        self.input_height: int = 224

        # 推理参数
        self.conf_threshold: float = 0.5
        self.num_classes: int = 1000
        self.top_k: int = 3

        # 推理设备
        self.device: str = "GPU"

        # 显示参数
        self.label_font_scale: float = 0.7
        self.label_color: Tuple[int, int, int] = (0, 255, 0)  # BGR

        # 结果缓存（供日志面板读取）
        self._result_summary: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像", data_type="图像")
        self.add_output("图像", data_type="图像")

    def _get_session(self):
        """加载 ONNX 会话，根据 device 选择推理后端。"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"模型文件不存在: {self.model_path}")
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self.device == "GPU":
            gpu_providers = [
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "DmlExecutionProvider",
            ]
            selected = [p for p in gpu_providers if p in available]
            providers = selected + ["CPUExecutionProvider"] if selected else ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        return ort.InferenceSession(self.model_path, providers=providers)

    def process(self, **inputs):
        img_rgb = inputs.get("图像")
        if img_rgb is None:
            raise ValueError("未接收到图像数据")

        session = self._get_session()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 预处理：缩放到模型输入尺寸
        blob = cv2.resize(img_bgr, (self.input_width, self.input_height),
                          interpolation=cv2.INTER_LINEAR)
        blob = blob.astype(np.float32) / 255.0

        # 均值/标准差归一化（ImageNet 标准）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob - mean) / std

        blob = np.transpose(blob, (2, 0, 1))          # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)            # -> NCHW

        # 推理
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        results = session.run([output_name], {input_name: blob})
        scores = results[0][0]  # [num_classes]

        # 找到 top_k 且高于阈值的类别
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        top_predictions = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] >= self.conf_threshold
        ]

        # 生成结果摘要供日志
        if top_predictions:
            self._result_summary = f"分类结果 (Top-{len(top_predictions)}):"
            for cls_id, score in top_predictions:
                self._result_summary += f"\n  #{cls_id}: {score:.4f}"
        else:
            self._result_summary = (
                f"无类别高于阈值 (threshold={self.conf_threshold})"
            )

        # 在图像左上角叠加分类结果
        drawn = self._draw_labels(img_bgr, top_predictions)

        result_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        return {"图像": result_rgb}

    def _draw_labels(self, img_bgr: np.ndarray,
                     predictions: List[Tuple[int, float]]) -> np.ndarray:
        """在图像左上角绘制分类标签。"""
        result = img_bgr.copy()
        color = self.label_color

        if not predictions:
            cv2.putText(result, "No detection above threshold",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        self.label_font_scale, (0, 0, 255), 2)
            return result

        y_offset = 25
        for cls_id, score in predictions:
            label = f"#{cls_id}: {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 2
            )
            # 背景
            cv2.rectangle(result, (6, y_offset - th - 4),
                          (tw + 16, y_offset + baseline + 2), color, -1)
            # 文字
            cv2.putText(result, label, (12, y_offset + baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale,
                        (0, 0, 0), 2)
            y_offset += th + baseline + 10

        return result
