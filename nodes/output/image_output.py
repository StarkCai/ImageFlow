"""图像输出算子：显示结果图像并支持保存到文件（含中文路径）。"""

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
    # cv2.imencode 期望 BGR 格式，此处图像为 RGB，需转换
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(filepath, "wb") as f:
        f.write(buf.tobytes())
    return True


class ImageViewerDialog(QDialog):
    """图像预览弹窗，图像自适应窗口大小，支持中文路径保存。"""

    def __init__(self, pixmap: QPixmap, raw_image: Optional[np.ndarray] = None,
                 title: str = "图像预览", parent=None):
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
        self._label.setSizePolicy(
            QLabel().sizePolicy().horizontalPolicy(),
            QLabel().sizePolicy().verticalPolicy(),
        )
        # 初始显示原始尺寸的图，由 resizeEvent 驱动缩放
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

        # 加载时立即缩放一次
        self._scale_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_pixmap()

    def _scale_pixmap(self):
        """将图像缩放到适应当前窗口大小，保持宽高比。"""
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
class ImageOutputNode(Node):
    display_name = "图像输出"
    category = "输出"

    def __init__(self):
        self._auto_show: bool = False
        self._save_dir: str = ""
        self._last_image: Optional[np.ndarray] = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("图像")

    def process(self, **inputs):
        img = inputs.get("图像")
        if img is None:
            raise ValueError("未接收到图像数据")

        self._last_image = img
        # 不在此处创建 GUI 对象 —— process() 可能在后台线程中运行，
        # Qt 禁止在非主线程创建 QWidget/QPixmap。
        return {"图像": img}

    def show_last_result(self):
        """手动显示上一次处理结果。"""
        if self._last_image is not None:
            img = self._last_image
            pixmap = _ndarray_to_qpixmap(img)
            viewer = ImageViewerDialog(pixmap, raw_image=img)
            viewer.exec_()

    def save_last_result(self, filepath: str):
        """保存上一次处理结果到文件，支持中文路径。"""
        if self._last_image is not None:
            _imwrite_unicode(filepath, self._last_image)
