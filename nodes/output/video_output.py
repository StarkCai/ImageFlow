"""视频/图像输出算子：支持将帧合成为视频，或逐帧保存为图像。"""

import os
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox,
)

from node_base import Node
from node_registry import register_node

_FOURCC_MAP = {
    "mp4": "mp4v",
    "avi": "XVID",
    "mov": "mp4v",
}


def _ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """将 numpy 图像数组转换为 QPixmap。"""
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
    """支持中文路径的图像保存，自动处理 RGB→BGR 转换。"""
    ext = os.path.splitext(filepath)[1]
    if not ext:
        ext = ".png"
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(filepath, "wb") as f:
        f.write(buf.tobytes())
    return True


class VideoViewerDialog(QDialog):
    """视频帧预览弹窗，支持图像保存。"""

    def __init__(self, pixmap: QPixmap, raw_image: Optional[np.ndarray] = None,
                 title: str = "帧预览", parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._raw = raw_image
        self._title = title
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)
        self.resize(900, 680)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background: #1e1e1e; border: none;")
        self._label.setPixmap(self._pixmap)
        layout.addWidget(self._label, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("保存图像")
        save_btn.setStyleSheet(
            "QPushButton { background: #5a9cf8; color: #fff; border: none; "
            "border-radius: 4px; padding: 6px 16px; font-size: 12px; }"
            "QPushButton:hover { background: #7ab4ff; }"
        )
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        self._scale_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_pixmap()

    def _scale_pixmap(self):
        if self._pixmap.isNull():
            return
        avail = self._label.size()
        if avail.width() <= 0 or avail.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            avail, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._label.setPixmap(scaled)

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;所有文件 (*)"
        )
        if path:
            if self._raw is not None:
                success = _imwrite_unicode(path, self._raw)
            else:
                success = self._pixmap.save(path)
            if not success:
                QMessageBox.warning(self, "保存失败", f"无法保存到 {path}")
            else:
                QMessageBox.information(self, "保存成功", f"图像已保存到:\n{path}")


@register_node
class VideoOutputNode(Node):
    display_name = "视频输出"
    category = "输出"

    def __init__(self):
        # ── 可序列化配置 ──
        self.output_mode: str = "video"     # "video" | "image"
        self.video_format: str = "mp4"      # "mp4", "avi", "mov"
        self.image_format: str = "jpg"      # "jpg", "png"
        self.save_dir: str = ""
        self.fps: int = 25
        self.auto_save: bool = False

        # ── 运行时状态 ──
        self._last_image: Optional[np.ndarray] = None
        self._accumulated_frames: list = []
        self._current_source_name: str = ""
        self._previous_source_name: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")

    def reset(self):
        """重写：保留帧缓冲和最后图像，仅清除引擎执行状态。"""
        self._data.clear()
        self._processed = False

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        self._last_image = img
        return {"图像": img}

    # ── 视频模式帧累积 ──────────────────────────────

    def append_frame(self, frame):
        """视频模式下累积帧（由引擎的 _post_batch_iteration 调用）。"""
        if frame is not None:
            self._accumulated_frames.append(frame.copy())

    def finalize_batch(self):
        """批处理结束后写入视频文件。"""
        if self.output_mode == "video" and self.auto_save and self._accumulated_frames:
            self._write_video()
            self._accumulated_frames.clear()

    def _write_video(self):
        """将累积的帧写入视频文件。"""
        if not self._accumulated_frames or not self.save_dir:
            return

        h, w = self._accumulated_frames[0].shape[:2]
        name = self._current_source_name or "output"
        ext = self.video_format
        output_path = os.path.join(self.save_dir, f"{name}.{ext}")

        os.makedirs(self.save_dir, exist_ok=True)

        fourcc_code = _FOURCC_MAP.get(ext, "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器: {output_path}")

        try:
            for frame in self._accumulated_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
        finally:
            writer.release()

    # ── 预览与手动保存 ──────────────────────────────

    def show_last_result(self):
        """手动显示上一次处理结果帧。"""
        if self._last_image is not None:
            pixmap = _ndarray_to_qpixmap(self._last_image)
            viewer = VideoViewerDialog(pixmap, raw_image=self._last_image)
            viewer.exec_()

    def save_last_result(self, filepath: str):
        """保存上一次处理结果到文件，支持中文路径。"""
        if self._last_image is not None:
            _imwrite_unicode(filepath, self._last_image)
