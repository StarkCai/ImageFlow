"""目标检测算子：基于 ONNX Runtime 通用推理框架，输入图像输出叠加框的图像和区域列表。"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from node_base import Node, format_regions
from node_registry import register_node


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """非极大值抑制，返回保留的索引列表。"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def _resize_keep_ratio(img: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, float, float, float]:
    """等比缩放并填充到目标尺寸，返回 (resized, ratio, pad_x, pad_y)。"""
    h, w = img.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, ratio, pad_x, pad_y


@register_node
class ObjectDetectionNode(Node):
    display_name = "目标检测"
    category = "深度学习"
    algorithms = ["目标检测"]

    def __init__(self):
        self.model_path: str = ""

        # 模型输入分辨率
        self.input_width: int = 640
        self.input_height: int = 640

        # 推理参数
        self.conf_threshold: float = 0.5
        self.iou_threshold: float = 0.45
        self.num_classes: int = 80

        # 推理设备
        self.device: str = "GPU"

        # 输出框配置
        self.box_type: str = "矩形"        # 矩形 / 圆形
        self.box_color: Tuple[int, int, int] = (0, 255, 0)  # BGR
        self.box_thickness: int = 2

        # YOLO 输出格式
        self.output_layout: str = "xyxy"       # xyxy / cxcywh
        self.coord_mode: str = "normalized"     # normalized (0~1) / pixel (输入分辨率坐标)

        # 结果缓存（供日志面板读取）
        self._result_summary: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像", data_type="图像")
        self.add_input("类别映射", data_type="类别映射")
        self.add_output("图像", data_type="图像")
        self.add_output("区域", data_type="区域")

    def _resolve_class_name(self, class_id: int, mapping: dict) -> str:
        """根据类别映射将 class_id 转为名称，无映射时返回 #id。"""
        if mapping:
            name = mapping.get(str(class_id)) or mapping.get(class_id)
            if name:
                return f"{name}"
        return f"#{class_id}"

    def _get_session(self):
        """加载 ONNX 会话，根据 device 选择推理后端。"""
        if not self.model_path or not os.path.exists(self.model_path):
            raise ValueError(f"模型文件不存在: {self.model_path}")
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self.device == "GPU":
            # 优先 GPU 后端
            gpu_providers = [
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "DmlExecutionProvider",
            ]
            selected = [p for p in gpu_providers if p in available]
            if selected:
                providers = selected + ["CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
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
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]

        # 预处理
        blob, ratio, pad_x, pad_y = _resize_keep_ratio(
            img_bgr, self.input_width, self.input_height
        )
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)     # -> NCHW

        # 自动获取模型输入输出名
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        # 推理
        results = session.run(output_names, {input_name: blob})
        detections = self._parse_output(results, orig_w, orig_h, ratio, pad_x, pad_y)

        # 绘制
        drawn = self._draw_detections(img_bgr, detections, class_mapping)

        # 转为标准区域格式
        regions_list = self._detections_to_regions(detections, class_mapping)
        regions = format_regions(regions_list)

        # 生成结果摘要供日志
        self._result_summary = (
            f"检测到 {len(detections)} 个目标"
            + (f" (阈值={self.conf_threshold}, IoU={self.iou_threshold})"
               if detections else "")
        )
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cls_name = self._resolve_class_name(d["class_id"], class_mapping)
            self._result_summary += (
                f"\n  {cls_name} conf={d['score']:.3f} "
                f"box=({x1},{y1})-({x2},{y2})"
            )

        result_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        return {"图像": result_rgb, "区域": regions}

    def _parse_output(self, results: list, orig_w: int, orig_h: int,
                      ratio: float, pad_x: float, pad_y: float) -> list:
        """解析模型输出为检测列表。"""

        def _to_original(x: float, y: float) -> Tuple[float, float]:
            """将模型输出坐标还原到原始图像坐标系。"""
            if self.coord_mode == "normalized":
                x = x * self.input_width
                y = y * self.input_height
            ox = (x - pad_x) / ratio
            oy = (y - pad_y) / ratio
            return (max(0.0, min(ox, orig_w)),
                    max(0.0, min(oy, orig_h)))

        def _box_to_original(x1: float, y1: float, x2: float, y2: float) -> list:
            ox1, oy1 = _to_original(x1, y1)
            ox2, oy2 = _to_original(x2, y2)
            return [ox1, oy1, ox2, oy2]

        # 尝试 YOLO 风格单输出 [1, N, 6]
        if len(results) == 1 and results[0].ndim == 3:
            preds = results[0][0]  # [N, 6]
            if preds.shape[1] < 6:
                return []
            detections = []
            for det in preds:
                if self.output_layout == "cxcywh":
                    cx, cy, bw, bh, conf, cls_id = det[:6]
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                else:
                    x1, y1, x2, y2, conf, cls_id = det[:6]

                if conf < self.conf_threshold:
                    continue
                cls_id = int(cls_id)
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                box = _box_to_original(x1, y1, x2, y2)
                detections.append({
                    "box": box,
                    "score": float(conf),
                    "class_id": cls_id,
                })

            if detections:
                boxes = np.array([d["box"] for d in detections])
                scores = np.array([d["score"] for d in detections])
                keep = _nms(boxes, scores, self.iou_threshold)
                return [detections[i] for i in keep]
            return []

        # 尝试多输出格式: boxes [1,N,4] + scores [1,N,C]
        array_outputs = [o for o in results if isinstance(o, np.ndarray) and o.ndim >= 2]
        if len(array_outputs) >= 2:
            boxes_out = array_outputs[0]
            scores_out = array_outputs[1]

            if boxes_out.ndim == 3:
                boxes_out = boxes_out[0]
            if scores_out.ndim == 3:
                scores_out = scores_out[0]

            if scores_out.shape[0] == self.num_classes and scores_out.shape[0] != boxes_out.shape[0]:
                scores_out = scores_out.T

            if boxes_out.shape[0] != scores_out.shape[0]:
                return []

            detections = []
            for i in range(boxes_out.shape[0]):
                if scores_out.shape[1] > 0:
                    cls_id = int(np.argmax(scores_out[i]))
                    conf = float(scores_out[i, cls_id])
                else:
                    conf = 1.0
                    cls_id = 0

                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = boxes_out[i][:4]
                box = _box_to_original(x1, y1, x2, y2)
                detections.append({
                    "box": box,
                    "score": conf,
                    "class_id": cls_id,
                })

            if detections:
                boxes = np.array([d["box"] for d in detections])
                scores = np.array([d["score"] for d in detections])
                keep = _nms(boxes, scores, self.iou_threshold)
                return [detections[i] for i in keep]

        return []

    def _draw_detections(self, img_bgr: np.ndarray, detections: list,
                         class_mapping: dict = None) -> np.ndarray:
        """在图像上绘制检测框。"""
        if class_mapping is None:
            class_mapping = {}
        result = img_bgr.copy()
        color = self.box_color
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            score = det["score"]
            cls_id = det["class_id"]
            cls_name = self._resolve_class_name(cls_id, class_mapping)

            if self.box_type == "圆形":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r = int(min(x2 - x1, y2 - y1) / 2)
                cv2.circle(result, (cx, cy), max(r, 1), color, self.box_thickness)
            else:
                cv2.rectangle(result, (x1, y1), (x2, y2), color, self.box_thickness)

            # 标签
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return result

    def _detections_to_regions(self, detections: list,
                               class_mapping: dict = None) -> List[dict]:
        """将检测结果转换为工程区域格式。"""
        if class_mapping is None:
            class_mapping = {}
        regions = []
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cls_id = det["class_id"]
            cls_name = self._resolve_class_name(cls_id, class_mapping)
            if self.box_type == "圆形":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r = int(min(x2 - x1, y2 - y1) / 2)
                regions.append({
                    "type": "圆形",
                    "class_id": cls_id,
                    "class": cls_name,
                    "coordinates": {"cx": cx, "cy": cy, "radius": max(r, 1)},
                })
            else:
                regions.append({
                    "type": "矩形",
                    "class_id": cls_id,
                    "class": cls_name,
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })
        return regions
