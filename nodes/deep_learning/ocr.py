"""字符识别 (OCR) 算子：基于 PP-OCR v5 ONNX 模型，检测文本区域并识别字符。

管线：
  1. 检测模型 (DBNet) → 文本区域概率图 → 二值化 → 轮廓提取 → 四点框
  2. 识别模型 (CRNN/SVTR) → 裁剪文本区域 → CTC 解码 → 字符序列
  3. 在白色背景上绘制文本区域和识别结果
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from node_base import Node, format_regions
from node_registry import register_node


def _min_area_rect_to_box(rect: tuple) -> np.ndarray:
    """将 cv2.minAreaRect 结果转为 4×2 点阵，按左上/右上/右下/左下排序。"""
    center, size, angle = rect
    w, h = size
    if w < h:
        w, h = h, w
        angle += 90
    # 取旋转矩形的四个顶点
    box = cv2.boxPoints(((center[0], center[1]), (w, h), angle))
    # 按 y 排序后分上下两组，每组按 x 排序
    box = box[np.argsort(box[:, 1])]
    top = box[:2][np.argsort(box[:2, 0])]
    bot = box[2:][np.argsort(box[2:, 0])[::-1]]
    return np.vstack([top, bot])


def _order_points(pts: np.ndarray) -> np.ndarray:
    """将四点排序为 左上 → 右上 → 右下 → 左下。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上（x+y 最小）
    rect[2] = pts[np.argmax(s)]   # 右下（x+y 最大）
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # 右上（y-x 最小）
    rect[3] = pts[np.argmax(d)]   # 左下（y-x 最大）
    return rect


def _crop_text_region(img: np.ndarray, box: np.ndarray,
                      target_h: int = 32) -> np.ndarray:
    """从图像中裁剪并矫正文本区域，缩放到识别模型输入尺寸。"""
    box = _order_points(box)
    src_w = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3]),
    ))
    src_h = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2]),
    ))
    if src_w <= 0 or src_h <= 0:
        return None

    dst_w = max(8, int(target_h * src_w / src_h))
    dst_pts = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, target_h - 1],
        [0, target_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst_pts)
    cropped = cv2.warpPerspective(img, M, (dst_w, target_h))
    return cropped


def _ctc_greedy_decode(probs: np.ndarray, chars: list) -> Tuple[str, float]:
    """CTC 贪心解码：取每步最大概率字符，去重并移除空白。返回 (text, confidence)。"""
    if probs.ndim != 2:
        return "", 0.0
    # probs: [T, C] — C 包含 blank (索引 0)
    token_ids = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    decoded = []
    total_conf = 0.0
    prev = -1
    for t, (tid, conf) in enumerate(zip(token_ids, confs)):
        if tid != prev and tid > 0 and tid < len(chars):
            decoded.append(chars[tid])
            total_conf += conf
        prev = tid

    text = "".join(decoded)
    if decoded:
        total_conf /= len(decoded)
    return text, float(total_conf)


def _load_dict(path: str) -> list:
    """加载字典文件（每行一个字符），索引 0 为 CTC blank。"""
    chars = [""]  # blank
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n\r")
            if c:
                chars.append(c)
    return chars


@register_node
class OcrNode(Node):
    display_name = "字符识别"
    category = "深度学习"
    algorithms = ["字符识别"]

    def __init__(self):
        # 检测模型
        self.det_model_path: str = ""
        self.det_input_size: int = 640

        # 识别模型
        self.rec_model_path: str = ""
        self.rec_input_height: int = 32

        # 字典
        self.dict_path: str = ""

        # 推理参数
        self.det_threshold: float = 0.3       # 检测二值化阈值
        self.det_box_thresh: float = 0.6      # 框置信度阈值
        self.det_unclip_ratio: float = 1.5    # 轮廓扩张比例
        self.rec_conf_threshold: float = 0.5  # 识别置信度阈值

        # 推理设备
        self.device: str = "GPU"

        # 显示
        self.text_color: Tuple[int, int, int] = (0, 0, 0)       # BGR
        self.box_color: Tuple[int, int, int] = (0, 0, 255)       # BGR (红色)
        self.text_scale: float = 0.6

        # 缓存
        self._result_summary: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像", data_type="图像")
        self.add_output("图像", data_type="图像")
        self.add_output("区域", data_type="区域")

    # ── 会话管理 ──────────────────────────────────────

    def _get_session(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
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

        return ort.InferenceSession(model_path, providers=providers)

    # ── 主流程 ────────────────────────────────────────

    def process(self, **inputs):
        img_rgb = inputs.get("图像")
        if img_rgb is None:
            raise ValueError("未接收到图像数据")

        # 加载字典
        chars = _load_dict(self.dict_path) if self.dict_path else []
        if not chars:
            raise ValueError("字典为空，请指定有效的字典文件")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]

        # ── 阶段 1：文本检测 ──
        if self.det_model_path:
            det_session = self._get_session(self.det_model_path)
            det_boxes = self._detect_text(img_bgr, orig_w, orig_h, det_session)
            mode_label = "检测+识别"
        else:
            # 仅识别模式：整图作为一个文本区域
            det_boxes = [
                np.array([[0, 0], [orig_w - 1, 0],
                          [orig_w - 1, orig_h - 1], [0, orig_h - 1]],
                         dtype=np.float32)
            ]
            mode_label = "仅识别(全图)"

        # ── 阶段 2：文本识别 ──
        rec_session = self._get_session(self.rec_model_path)
        # 从模型输入形状自动检测识别高度（默认 fallback 32）
        rec_height = self._detect_rec_height(rec_session)
        results = self._recognize_text(img_bgr, det_boxes, rec_session, chars, rec_height)

        # ── 阶段 3：绘制结果 ──
        # 仅识别模式：文字叠加到原图；检测+识别模式：白底 + 检测框
        if not self.det_model_path:
            drawn = self._draw_overlay(img_bgr, results)
        else:
            drawn = self._draw_results(img_bgr.shape, results)

        # 日志摘要
        self._result_summary = f"[{mode_label}] 识别到 {len(results)} 个文本"
        for r in results:
            self._result_summary += (
                f"\n  \"{r['ocr']}\" conf={r['rec_conf']:.3f}"
            )

        result_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        regions_list = [
            {
                "type": "多边形",
                "ocr": r["ocr"],
                "class_id": None,
                "coordinates": {
                    "points": [[int(x), int(y)] for x, y in r["box"]]
                },
            }
            for r in results
        ]
        return {"图像": result_rgb, "区域": format_regions(regions_list, width=orig_w, height=orig_h)}

    # ── 文本检测 ──────────────────────────────────────

    def _detect_text(self, img_bgr: np.ndarray, orig_w: int, orig_h: int,
                     session) -> list:
        """DBNet 检测：推理 → 概率图 → 二值化 → 轮廓 → 四点框。"""
        # 预处理：等比缩放并填充
        ratio = min(self.det_input_size / orig_w, self.det_input_size / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到 32 的倍数
        pad_w = (32 - new_w % 32) % 32
        pad_h = (32 - new_h % 32) % 32
        canvas = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = resized.astype(np.float32)

        # 归一化：[0, 255] → [0, 1]
        blob = canvas / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # 推理
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: blob})

        prob_map = self._parse_det_output(outputs, canvas.shape[:2])

        # 缩放到画布分辨率
        prob_map = cv2.resize(prob_map, (new_w + pad_w, new_h + pad_h))

        # 二值化
        mask = (prob_map > self.det_threshold).astype(np.uint8) * 255

        # 轮廓提取
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if len(cnt) < 4:
                continue
            area = cv2.contourArea(cnt)
            if area < 16:
                continue

            # 扩张轮廓（DBNet 的 unclip 等价操作）
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < 4:
                continue

            # 最小外接矩形 → 四点框
            rect = cv2.minAreaRect(approx)
            box = _min_area_rect_to_box(rect)

            # 缩放回原始图像坐标
            box = box / ratio
            box = np.clip(box, [0, 0], [orig_w - 1, orig_h - 1])

            # 框最小尺寸过滤
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[0] - box[3])
            if min(w, h) < 6:
                continue

            # 框置信度（区域内概率均值）
            bx, by, bw, bh = cv2.boundingRect(box.astype(np.int32))
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(prob_map.shape[1] - bx, bw), min(prob_map.shape[0] - by, bh)
            if bw > 0 and bh > 0:
                # 映射回画布坐标计算置信度
                canvas_bx = int(bx * ratio)
                canvas_by = int(by * ratio)
                canvas_bw = int(bw * ratio)
                canvas_bh = int(bh * ratio)
                canvas_bx = max(0, min(canvas_bx, prob_map.shape[1] - 1))
                canvas_by = max(0, min(canvas_by, prob_map.shape[0] - 1))
                canvas_bw = max(1, min(canvas_bw, prob_map.shape[1] - canvas_bx))
                canvas_bh = max(1, min(canvas_bh, prob_map.shape[0] - canvas_by))
                roi = prob_map[canvas_by:canvas_by + canvas_bh,
                               canvas_bx:canvas_bx + canvas_bw]
                box_conf = float(np.mean(roi)) if roi.size > 0 else 0.0
            else:
                box_conf = 0.0

            if box_conf < self.det_box_thresh:
                continue

            boxes.append(box)

        # 按 y 坐标排序（从上到下，从左到右）
        boxes.sort(key=lambda b: (b[:, 1].mean(), b[:, 0].mean()))
        return boxes

    def _parse_det_output(self, outputs: list, canvas_shape: tuple) -> np.ndarray:
        """解析检测模型输出为概率图 [H, W]。"""
        canvas_h, canvas_w = canvas_shape

        # 尝试找到最可能是概率图的输出（值在 [0,1] 区间的 2D/3D 特征图）
        for out in outputs:
            if isinstance(out, np.ndarray):
                arr = out
                # 去掉 batch 和 channel 维度
                while arr.ndim > 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[0]
                    else:
                        break
                if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
                    # 缩放到画布尺寸
                    if arr.shape != (canvas_h, canvas_w):
                        arr = cv2.resize(arr.astype(np.float32),
                                         (canvas_w, canvas_h))
                    return arr.astype(np.float32)

        # 回退：取最后一个合适的输出
        for out in reversed(outputs):
            if isinstance(out, np.ndarray) and out.ndim >= 2:
                arr = out
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.ndim == 2:
                    return cv2.resize(arr.astype(np.float32),
                                      (canvas_w, canvas_h))
        return np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # ── 文本识别 ──────────────────────────────────────

    def _detect_rec_height(self, session) -> int:
        """从识别模型输入形状中检测期望的高度。"""
        try:
            input_info = session.get_inputs()[0]
            shape = input_info.shape  # e.g. [-1, 3, 48, -1] or [1, 3, 32, 100]
            if len(shape) >= 3:
                h = shape[2]
                if isinstance(h, int) and h > 0:
                    return h
        except Exception:
            pass
        return self.rec_input_height  # fallback

    def _recognize_text(self, img_bgr: np.ndarray, boxes: list,
                        session, chars: list, rec_height: int = 32) -> List[dict]:
        """对每个检测框执行文本识别。"""
        results = []
        for box in boxes:
            # 裁剪并矫正文本区域
            cropped = _crop_text_region(img_bgr, box, rec_height)
            if cropped is None or cropped.size == 0:
                continue

            # 预处理
            blob = cropped.astype(np.float32) / 255.0
            blob = (blob - 0.5) / 0.5  # [-1, 1]
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            # 推理
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
            outputs = session.run(output_names, {input_name: blob})

            probs = self._parse_rec_output(outputs, chars)
            if probs is None:
                continue

            text, conf = _ctc_greedy_decode(probs, chars)
            if not text or conf < self.rec_conf_threshold:
                continue

            results.append({
                "box": box,
                "ocr": text,
                "rec_conf": conf,
            })

        return results

    def _parse_rec_output(self, outputs: list, chars: list) -> Optional[np.ndarray]:
        """解析识别模型输出为 CTC 概率矩阵 [T, C]。"""
        for out in outputs:
            if isinstance(out, np.ndarray):
                arr = out
                # 去除 batch 和多余的 1 维度
                while arr.ndim > 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]
                    else:
                        break
                if arr.ndim == 2:
                    # [T, C] 或 [C, T]
                    # 概率矩阵的 C 维应接近字符表大小
                    if arr.shape[1] >= len(chars) - 10 and arr.shape[1] <= len(chars) + 10:
                        return arr.astype(np.float64)
                    elif arr.shape[0] >= len(chars) - 10 and arr.shape[0] <= len(chars) + 10:
                        return arr.T.astype(np.float64)
        return None

    # ── 绘制 ──────────────────────────────────────────

    @staticmethod
    def _get_cjk_font(size: int = 20):
        """查找可用的中文字体，返回 PIL ImageFont。"""
        from PIL import ImageFont
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei
            "C:/Windows/Fonts/simhei.ttf",     # SimHei
            "C:/Windows/Fonts/simsun.ttc",     # SimSun
            "C:/Windows/Fonts/msyhbd.ttc",     # Microsoft YaHei Bold
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _draw_overlay(self, img_bgr: np.ndarray, results: list) -> np.ndarray:
        """仅识别模式：将文字自适应大小叠加到原图上（半透明深色底 + 白字）。"""
        from PIL import Image, ImageDraw

        h, w = img_bgr.shape[:2]
        font_size = max(18, min(int(h * 0.06), 56))
        font = self._get_cjk_font(font_size)

        # 原图 → PIL RGBA
        bg_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(bg_rgb).convert("RGBA")

        # 文字叠加层（全透明）
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        y_offset = max(10, h // 6)
        for r in results:
            text = r["ocr"]
            conf = r["rec_conf"]
            label = f"  {text}  ({conf:.2f})  "

            bbox = draw_overlay.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # 居中靠上
            tx = max(10, (w - tw) // 2)
            ty = y_offset

            pad = 10
            # 半透明深色背景 + 白色文字
            draw_overlay.rectangle(
                [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                fill=(0, 0, 0, 170),
            )
            draw_overlay.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

            y_offset += th + pad * 2 + 8

        # 合成 → BGR
        pil_img = Image.alpha_composite(pil_img, overlay)
        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _draw_results(self, img_shape: tuple, results: list) -> np.ndarray:
        """在白色背景上绘制识别结果（PIL 渲染中文）。"""
        from PIL import Image, ImageDraw

        h, w = img_shape[:2]
        # 根据图像尺寸动态计算字体大小
        font_size = max(14, min(int(h * 0.025), 36))
        font = self._get_cjk_font(font_size)

        # 创建白色 PIL 画布
        pil_img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(pil_img)

        # 线条宽度
        line_w = max(1, int(min(w, h) * 0.0015))

        for r in results:
            box = r["box"]
            text = r["ocr"]
            conf = r["rec_conf"]

            # 多边形顶点坐标 (int tuples)
            pts = [(int(x), int(y)) for x, y in box]

            # 绘制文本区域框（红色）
            draw.line(pts + [pts[0]], fill=(255, 0, 0), width=line_w)

            # 计算文字位置（框左上角上方）
            tx = pts[0][0]
            ty = pts[0][1] - 6
            if ty < font_size + 4:
                ty = max(p[1] for p in pts) + 6

            label = f"{text} ({conf:.2f})"

            # PIL 文字背景
            bbox = draw.textbbox((tx, ty), label, font=font, anchor="la")
            # 背景矩形
            pad = 2
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=(255, 0, 0),
            )
            draw.text((tx, ty), label, font=font, fill=(0, 0, 0), anchor="la")

        # PIL → numpy BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
