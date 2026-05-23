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
        self.conf_threshold: float = 0.0
        self.num_classes: int = 1000
        self.top_k: int = 1

        # 推理设备
        self.device: str = "GPU"

        # 显示参数
        self.label_font_scale: float = 0.7

        # 结果缓存（供日志面板读取）
        self._result_summary: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像", data_type="图像")
        self.add_input("类别映射", data_type="类别映射")
        self.add_output("图像", data_type="图像")

    def _resolve_class_name(self, class_id: int, mapping: dict) -> str:
        """根据类别映射将 class_id 转为名称，无映射时返回 #id。"""
        if mapping:
            name = mapping.get(str(class_id)) or mapping.get(class_id)
            if name:
                return name
        return f"#{class_id}"

    @staticmethod
    def _class_color(class_id: int) -> Tuple[int, int, int]:
        """为每个类别 ID 生成区分度高的 BGR 颜色（黄金比例色相分布）。"""
        hue = (class_id * 0.618033988749895) % 1.0
        h = hue * 6.0
        c = 0.9 * 0.75
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c
        if h < 1:   r, g, b = c, x, 0
        elif h < 2: r, g, b = x, c, 0
        elif h < 3: r, g, b = 0, c, x
        elif h < 4: r, g, b = 0, x, c
        elif h < 5: r, g, b = x, 0, c
        else:       r, g, b = c, 0, x
        return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))

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

        # 解析类别映射
        class_mapping = {}
        raw_mapping = inputs.get("类别映射")
        if raw_mapping and isinstance(raw_mapping, dict):
            class_mapping = raw_mapping.get("mapping", {})

        session = self._get_session()

        # ── CUDA 状态检查（全局仅一次）────────────
        import node_base as _nb
        if not _nb._cuda_status_logged:
            providers = session.get_providers()
            _nb._cuda_status_logged = True
            using_cuda = "CUDAExecutionProvider" in providers
            self._cuda_log_msg = "[CUDA 加速已启用]" if using_cuda else "[CUDA 不可用，使用 CPU 推理]"
            print(f"[Image Flow] {self._cuda_log_msg}")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 预处理：缩放到模型输入尺寸
        blob = cv2.resize(img_bgr, (self.input_width, self.input_height),
                          interpolation=cv2.INTER_LINEAR)
        blob = blob.astype(np.float32) / 255.0

        # ImageNet 均值/标准差归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob - mean) / std

        blob = np.transpose(blob, (2, 0, 1))          # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)            # -> NCHW

        # 推理
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        results = session.run(output_names, {input_name: blob})

        # 解析输出分数
        scores = self._parse_scores(results)

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
                cls_name = self._resolve_class_name(cls_id, class_mapping)
                self._result_summary += f"\n  {cls_name}: {score:.4f}"
        else:
            self._result_summary = (
                f"无类别高于阈值 (threshold={self.conf_threshold})"
            )

        # 在图像左上角叠加分类结果
        drawn = self._draw_labels(img_bgr, top_predictions, class_mapping)

        result_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        return {"图像": result_rgb}

    def _parse_scores(self, results: list) -> np.ndarray:
        """从模型输出中解析类别分数，自动检测并应用 softmax。"""
        out = results[0]

        # 去除 batch 维度
        if out.ndim == 4:
            out = out[0, :, 0, 0]
        elif out.ndim == 2:
            out = out[0]
        elif out.ndim == 1:
            pass
        else:
            out = out.flatten()

        scores = out.astype(np.float64)

        # 去除 batch 维后可能仍是 2D (如 [1, N])
        if scores.ndim == 2:
            scores = scores[0]

        # 裁剪或填充到 num_classes
        if len(scores) > self.num_classes:
            scores = scores[:self.num_classes]
        elif len(scores) < self.num_classes:
            padded = np.zeros(self.num_classes, dtype=np.float64)
            padded[:len(scores)] = scores
            scores = padded

        # 自动检测是否需要 softmax：若总和远离 1 或存在负值，视为 logits
        if scores.sum() < 0.9 or scores.sum() > 1.1 or scores.min() < 0:
            scores = np.exp(scores - np.max(scores))
            scores = scores / scores.sum()

        return scores

    def _draw_labels(self, img_bgr: np.ndarray,
                     predictions: List[Tuple[int, float]],
                     class_mapping: dict = None) -> np.ndarray:
        """在图像左上角绘制分类标签，不同类别使用不同背景色。"""
        if class_mapping is None:
            class_mapping = {}
        result = img_bgr.copy()

        if not predictions:
            cv2.putText(result, "No detection above threshold",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        self.label_font_scale, (0, 0, 255), 2)
            return result

        y_offset = 25
        for cls_id, score in predictions:
            cls_name = self._resolve_class_name(cls_id, class_mapping)
            color = self._class_color(cls_id)

            label = f"{cls_name}: {score:.2f}"
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
