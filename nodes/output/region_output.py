"""区域输出算子：可视化预览区域 JSON，支持弹窗详情与下载（含中文路径）。"""

import json
import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFileDialog, QMessageBox,
)

from node_base import Node
from node_registry import register_node


class JsonViewerDialog(QDialog):
    """JSON 详情弹窗，等宽字体展示格式化 JSON。"""

    def __init__(self, json_text: str, title: str = "区域数据", parent=None):
        super().__init__(parent)
        self._json_text = json_text
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        self.resize(700, 550)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setFont(QFont("Consolas", 11))
        editor.setStyleSheet(
            "QTextEdit { background: #15151a; color: #c8c8d0; "
            "border: 1px solid #333; border-radius: 4px; padding: 8px; }"
        )
        editor.setPlainText(self._json_text)
        layout.addWidget(editor)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("保存 JSON")
        save_btn.setStyleSheet(
            "QPushButton { background: #5a9cf8; color: #fff; border: none; "
            "border-radius: 4px; padding: 6px 16px; font-size: 12px; }"
            "QPushButton:hover { background: #7ab4ff; }"
        )
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(
            "QPushButton { background: #3d3d45; color: #dcdce0; border: 1px solid #555; "
            "border-radius: 4px; padding: 6px 16px; font-size: 12px; }"
            "QPushButton:hover { background: #4d4d58; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存区域 JSON", "regions.json",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._json_text)
            QMessageBox.information(self, "保存成功", f"JSON 已保存到:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))


@register_node
class RegionOutputNode(Node):
    display_name = "区域输出"
    category = "输出"

    def __init__(self):
        self._last_regions: Optional[dict] = None
        self._last_json: str = ""
        self.save_json: bool = False
        self.save_dir: str = ""
        super().__init__()

    def _setup_ports(self):
        self.add_input("区域", data_type="区域")

    def process(self, **inputs):
        data = inputs.get("区域")
        if data is None:
            raise ValueError("未接收到区域数据")
        self._last_regions = data
        self._last_json = json.dumps(data, ensure_ascii=False, indent=2)
        return {"区域": data}

    @property
    def region_summary(self) -> str:
        if self._last_regions is None:
            return "暂无区域数据"
        regions = self._last_regions.get("regions", [])
        count = len(regions)
        if count == 0:
            return "区域为空"
        types = {}
        for r in regions:
            t = r.get("type", "未知")
            types[t] = types.get(t, 0) + 1
        type_str = ", ".join(f"{t}×{c}" for t, c in types.items())
        return f"共 {count} 个区域 ({type_str})"

    def show_detail(self):
        if not self._last_json:
            QMessageBox.information(None, "提示", "暂无区域数据，请先执行流程。")
            return
        dlg = JsonViewerDialog(self._last_json, "区域数据详情")
        dlg.exec_()

    def save_json(self):
        if not self._last_json:
            QMessageBox.information(None, "提示", "暂无区域数据，请先执行流程。")
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "保存区域 JSON", "regions.json",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return
        self._write_json(path)

    def save_last_json(self, filepath: str):
        """批处理自动保存 JSON 到指定文件（支持中文路径），静默忽略错误。"""
        if self._last_json:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self._last_json)
            except Exception:
                pass

    def _write_json(self, filepath: str):
        """手动保存 JSON，弹出错误提示。"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self._last_json)
        except Exception as e:
            QMessageBox.warning(None, "保存失败", str(e))
