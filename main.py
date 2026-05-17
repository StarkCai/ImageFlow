"""
图像处理流式编辑器 —— 主启动脚本。

用法:
    python main.py

依赖安装:
    pip install -r requirements.txt
"""

from __future__ import annotations
import sys
import os
import traceback
from typing import Type, Optional

from PyQt5.QtCore import Qt, QMimeData, QPointF, QSize, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QDrag, QPixmap, QPainter, QPen, QBrush, QIcon,
    QTextCursor,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QToolBar, QAction, QStatusBar, QSizePolicy, QCheckBox, QMenuBar,
    QMenu, QTextEdit, QRadioButton, QButtonGroup,
)

# 必须在导入节点之前设置 Qt 平台插件路径
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from node_base import Node, Port, Connection, ExecutionEngine
from node_registry import NodeRegistry
from node_canvas import NodeScene, NodeCanvas, NodeItem, WireItem
from nodes.output.image_output import _ndarray_to_qpixmap
from project import save_project, load_project, apply_params

# 触发节点注册
import nodes  # noqa: F401

# ── 样式常量 ──────────────────────────────────────────
SIDEBAR_BG = "#2b2b30"
SIDEBAR_TEXT = "#dcdce0"
SIDEBAR_DIM = "#9a9aa0"
ACCENT = "#5a9cf8"
ACCENT_HOVER = "#7ab4ff"
BTN_BG = "#3d3d45"
BTN_HOVER = "#4d4d58"
DANGER = "#e05560"
SUCCESS = "#5cb878"


# ── 异步流程执行器 ────────────────────────────────────
class FlowRunner(QThread):
    """在后台线程中执行节点流程，避免阻塞 UI 并防止算法错误导致主界面崩溃。"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    node_started = pyqtSignal(str)
    node_done = pyqtSignal(str)
    node_error = pyqtSignal(str, str)     # node_name, error_msg
    batch_progress = pyqtSignal(int, int, str)  # current, total, filename

    def __init__(self, engine: ExecutionEngine):
        super().__init__()
        self.engine = engine
        self.engine._progress_callback = self._on_node_progress
        self._progress_counter = 0
        self._progress_interval = 5  # 每 5 帧发射一次进度信号

    def _on_node_progress(self, node_name: str, phase: str):
        if phase == "start":
            self.node_started.emit(node_name)
        elif phase == "done":
            self.node_done.emit(node_name)
        elif phase.startswith("error:"):
            self.node_error.emit(node_name, phase[7:])
        elif phase.startswith("start_image_"):
            self._progress_counter += 1
            if self._progress_counter % self._progress_interval != 0:
                return
            # 格式: "start_image_{idx}_{total}_{filename}"
            parts = phase.split("_", 3)
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    total = int(parts[2])
                    filename = parts[3] if len(parts) > 3 else ""
                    self.batch_progress.emit(idx, total, filename)
                except ValueError:
                    pass

    def run(self):
        try:
            results = self.engine.execute_batch()
            self.finished.emit(results)
        except Exception:
            self.error.emit(traceback.format_exc())

    def cancel(self):
        self.engine.cancel()


# ── 可拖拽的节点列表 ─────────────────────────────────
class NodeListItem(QWidget):
    """节点面板中的每一项，支持拖拽。"""

    def __init__(self, node_cls: Type[Node], parent=None):
        super().__init__(parent)
        self.node_cls = node_cls
        self.setToolTip(f"{node_cls.category} -> {node_cls.display_name}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)

        # 类别色标
        color_map = {"输入": ACCENT, "图像处理": SUCCESS, "输出": "#f0a050", "图像叠加": "#c084fc", "深度学习": "#f472b6"}
        dot = QLabel("●")
        dot.setStyleSheet(
            f"color: {color_map.get(node_cls.category, ACCENT)}; "
            f"font-size: 12px; background: transparent;"
        )
        dot.setFixedWidth(16)
        layout.addWidget(dot)

        label = QLabel(node_cls.display_name)
        label.setStyleSheet(f"color: {SIDEBAR_TEXT}; font-size: 12px; background: transparent;")
        layout.addWidget(label)

        layout.addStretch()

        cat = QLabel(node_cls.category)
        cat.setStyleSheet(f"color: {SIDEBAR_DIM}; font-size: 10px; background: transparent;")
        layout.addWidget(cat)

        self.setStyleSheet(
            f"NodeListItem {{ background: {BTN_BG}; border-radius: 6px; "
            f"border: 1px solid transparent; }}"
            f"NodeListItem:hover {{ background: {BTN_HOVER}; border-color: #555; }}"
        )
        self.setFixedHeight(36)
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not hasattr(self, "_drag_start"):
            return
        if (event.pos() - self._drag_start).manhattanLength() < 10:
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(self.node_cls.display_name)
        mime.setData("application/x-node-type", self.node_cls.display_name.encode())
        drag.setMimeData(mime)

        # 拖拽缩略图
        pixmap = QPixmap(140, 50)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(85, 85, 95))
        painter.setPen(QPen(QColor(110, 110, 125), 1.5))
        painter.drawRoundedRect(2, 2, 136, 46, 8, 8)
        painter.setPen(QColor(220, 220, 225))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(10, 28, self.node_cls.display_name)
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPointF(70, 25).toPoint())
        drag.exec_(Qt.CopyAction)


class CollapsibleSection(QWidget):
    """可折叠分组控件，点击标题栏展开/折叠内容区域。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._title = title
        self._build_ui()

    def _build_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        # 标题栏
        self._header = QPushButton()
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setFixedHeight(32)
        self._header.setStyleSheet(
            f"QPushButton {{ background: {BTN_HOVER}; border: none; border-radius: 4px; "
            f"text-align: left; padding: 6px 10px; color: {SIDEBAR_TEXT}; "
            f"font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #555560; }}"
        )
        self._header.clicked.connect(self._toggle)
        self._layout.addWidget(self._header)

        # 内容容器
        self._content = QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 2, 0, 6)
        self._content_layout.setSpacing(2)
        self._content.setVisible(False)
        self._layout.addWidget(self._content)

        self._update_header_text()

    def _toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._update_header_text()

    def _update_header_text(self):
        arrow = "▼" if self._expanded else "▶"
        self._header.setText(f"  {arrow}  {self._title}")

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)


class NodePanel(QWidget):
    """左侧节点列表面板。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)

        title = QLabel("算子列表")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setStyleSheet(f"color: {SIDEBAR_TEXT}; padding: 4px 0; background: transparent;")
        layout.addWidget(title)

        hint = QLabel("拖拽算子到画布 或 双击添加")
        hint.setStyleSheet(f"color: {SIDEBAR_DIM}; font-size: 10px; background: transparent;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addSpacing(8)

        registry = NodeRegistry()
        grouped = registry.list_by_category()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 3px; }}"
        )

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        self._list_items: dict[str, NodeListItem] = {}

        for cat_name, node_classes in grouped.items():
            section = CollapsibleSection(cat_name)
            container_layout.addWidget(section)

            for cls in node_classes:
                item = NodeListItem(cls)
                item.mouseDoubleClickEvent = lambda e, c=cls: self._on_double_click(c)
                section.add_widget(item)
                self._list_items[cls.display_name] = item

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _on_double_click(self, node_cls: Type[Node]):
        """双击默认将节点添加到画布中心。"""
        main_win = self.window()
        if isinstance(main_win, MainWindow):
            main_win.add_node_to_center(node_cls)


# ── 日志面板 ──────────────────────────────────────────
LOG_BG = "#15151a"
LOG_TEXT = "#b8b8c0"
LOG_INFO = "#7eb8da"
LOG_WARN = "#e0a050"
LOG_ERROR = "#e05560"
LOG_SUCCESS = "#5cb878"
LOG_TIMESTAMP = "#5a5a6a"


class LogPanel(QWidget):
    """底部日志面板，线程安全地显示算子运行日志和错误信息。"""

    _append_signal = pyqtSignal(str, str)  # level, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._append_signal.connect(self._do_append)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # 标题栏
        header = QHBoxLayout()
        title = QLabel("执行日志")
        title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        title.setStyleSheet(f"color: {SIDEBAR_DIM}; background: transparent;")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("清空")
        clear_btn.setFixedSize(50, 20)
        clear_btn.setStyleSheet(
            f"QPushButton {{ background: #3d3d45; color: {SIDEBAR_DIM}; "
            f"border: none; border-radius: 3px; font-size: 10px; }}"
            f"QPushButton:hover {{ background: #555; color: {SIDEBAR_TEXT}; }}"
        )
        clear_btn.clicked.connect(self.clear)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        # 日志文本区
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._text.setStyleSheet(
            f"QTextEdit {{ background: {LOG_BG}; color: {LOG_TEXT}; "
            f"border: 1px solid #333; border-radius: 4px; "
            f"font-size: 11px; font-family: 'Consolas', 'Microsoft YaHei'; "
            f"padding: 4px 6px; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: #444; border-radius: 3px; }}"
        )
        layout.addWidget(self._text)

    def info(self, msg: str):
        self._append_signal.emit("INFO", msg)

    def success(self, msg: str):
        self._append_signal.emit("OK", msg)

    def warn(self, msg: str):
        self._append_signal.emit("WARN", msg)

    def error(self, msg: str):
        self._append_signal.emit("ERROR", msg)

    def clear(self):
        self._text.clear()

    def _do_append(self, level: str, message: str):
        ts = QDateTime.currentDateTime().toString("HH:mm:ss")
        color_map = {
            "INFO": LOG_INFO,
            "OK": LOG_SUCCESS,
            "WARN": LOG_WARN,
            "ERROR": LOG_ERROR,
        }
        color = color_map.get(level, LOG_TEXT)

        self._text.moveCursor(QTextCursor.End)
        self._text.insertHtml(
            f'<span style="color:{LOG_TIMESTAMP};">[{ts}]</span> '
            f'<span style="color:{color}; font-weight:bold;">[{level}]</span> '
            f'<span style="color:{LOG_TEXT};">{message}</span><br>'
        )
        # 自动滚动到底部
        self._text.moveCursor(QTextCursor.End)


# ── 属性面板 ──────────────────────────────────────────
class PropertyPanel(QWidget):
    """右侧属性编辑面板。
    上半区域：算法选择（仅多算法节点显示）
    下半区域：算子参数配置。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_node: Optional[Node] = None
        self._current_item: Optional[NodeItem] = None
        self._form_layout: Optional[QFormLayout] = None
        self._preview_label: Optional[QLabel] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(4)

        title = QLabel("属性面板")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setStyleSheet(f"color: {SIDEBAR_TEXT}; padding: 4px 0; background: transparent;")
        layout.addWidget(title)

        self._hint = QLabel("选择一个节点以编辑参数")
        self._hint.setStyleSheet(f"color: {SIDEBAR_DIM}; font-size: 10px; background: transparent;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        # 上半区域：算法选择
        algo_title = QLabel("算法选择")
        algo_title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        algo_title.setStyleSheet(f"color: {ACCENT}; padding: 4px 0; background: transparent;")
        layout.addWidget(algo_title)
        self._algo_title = algo_title

        self._algo_combo = QComboBox()
        self._algo_combo.setStyleSheet(self._input_style())
        self._algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self._algo_combo)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("background: #444;")
        layout.addWidget(sep1)
        self._algo_sep = sep1

        # 下半区域：算子参数
        param_title = QLabel("算子参数")
        param_title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        param_title.setStyleSheet(f"color: {ACCENT}; padding: 4px 0; background: transparent;")
        layout.addWidget(param_title)
        self._param_title = param_title

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 3px; }}"
        )
        self._scroll = scroll

        self._form_container = QWidget()
        self._form_container.setStyleSheet("background: transparent;")
        self._form_container.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self._form_layout = QFormLayout(self._form_container)
        self._form_layout.setContentsMargins(4, 4, 4, 4)
        self._form_layout.setSpacing(8)
        self._form_layout.setLabelAlignment(Qt.AlignLeft)

        scroll.setWidget(self._form_container)
        layout.addWidget(scroll, stretch=1)

        self._set_algo_area_visible(False)

    def _set_algo_area_visible(self, visible: bool):
        self._algo_title.setVisible(visible)
        self._algo_combo.setVisible(visible)
        self._algo_sep.setVisible(visible)

    def _on_algorithm_changed(self, text: str):
        if self._current_node is None:
            return
        if not hasattr(self._current_node, "algorithm"):
            return
        self._current_node.algorithm = text
        self._rebuild_params()

    def _rebuild_params(self):
        """重建参数区域（算法切换时调用）。"""
        node = self._current_node
        if node is None:
            return
        self._clear_params()
        self._add_node_params(node)

    def set_node(self, node: Optional[Node], item: Optional[NodeItem] = None):
        """加载节点属性到面板。"""
        self._current_node = node
        self._current_item = item
        self._clear_params()

        if node is None:
            self._hint.setText("选择一个节点以编辑参数")
            self._hint.setVisible(True)
            self._set_algo_area_visible(False)
            return

        self._hint.setVisible(False)

        # 算法选择区域（仅多算法节点）
        algos = getattr(node, "algorithms", None)
        if algos and len(algos) > 1:
            self._algo_combo.blockSignals(True)
            self._algo_combo.clear()
            self._algo_combo.addItems(algos)
            current = getattr(node, "algorithm", algos[0])
            self._algo_combo.setCurrentText(current)
            self._algo_combo.blockSignals(False)
            self._set_algo_area_visible(True)
        else:
            self._set_algo_area_visible(False)

        # 节点标题
        title_label = QLabel(f"{node.display_name}")
        title_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        self._form_layout.addRow(title_label)

        # 参数
        self._add_node_params(node)

    def _clear_params(self):
        """清空参数区域。"""
        self._preview_label = None
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)
        for child in self._form_container.findChildren(QWidget):
            child.deleteLater()

    def refresh_output_preview(self):
        """执行流程后刷新输出节点的预览图像。"""
        node = self._current_node
        if node is None or not hasattr(node, "_last_image"):
            return
        preview = getattr(self, "_preview_label", None)
        if preview is None or node._last_image is None:
            return
        pixmap = _ndarray_to_qpixmap(node._last_image)
        scaled = pixmap.scaled(
            180, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        preview.setPixmap(scaled)

    def _refresh_region_output_from_nodes(self, nodes):
        """执行后刷新当前选中节点（如果是 RegionOutputNode）的信息。"""
        node = self._current_node
        if node is None:
            # 尝试从执行节点中找 RegionOutputNode 刷新
            for n in nodes:
                if n.__class__.__name__ == "RegionOutputNode":
                    self._refresh_region_output(n)
            return
        if node.__class__.__name__ == "RegionOutputNode":
            self._refresh_region_output(node)

    def _add_node_params(self, node: Node):
        """根据节点类型动态创建参数控件。"""
        cls_name = node.__class__.__name__

        if cls_name == "ImageInputNode":
            self._add_image_input_params(node)
        elif cls_name == "VideoInputNode":
            self._add_video_input_params(node)
        elif cls_name == "SmoothingNode":
            self._add_smoothing_params(node)
        elif cls_name == "EdgeDetectNode":
            self._add_edge_params(node)
        elif cls_name == "ThresholdNode":
            self._add_threshold_params(node)
        elif cls_name == "GeometryTransformNode":
            self._add_geometry_params(node)
        elif cls_name == "MorphologyNode":
            self._add_morphology_params(node)
        elif cls_name == "EnhancementNode":
            self._add_enhancement_params(node)
        elif cls_name == "FrequencyNode":
            self._add_frequency_params(node)
        elif cls_name == "SegmentationNode":
            self._add_segmentation_params(node)
        elif cls_name == "FeatureDetectionNode":
            self._add_feature_detection_params(node)
        elif cls_name == "ColorSpaceNode":
            pass  # 颜色空间转换无需额外参数，算法即转换类型
        elif cls_name == "RegionInputNode":
            self._add_region_input_params(node)
        elif cls_name == "ClassMappingNode":
            self._add_class_mapping_params(node)
        elif cls_name == "OverlayNode":
            self._add_overlay_params(node)
        elif cls_name == "CropNode":
            pass  # 算法选择由 PropertyPanel 自动处理，无额外参数
        elif cls_name == "ObjectDetectionNode":
            self._add_object_detection_params(node)
        elif cls_name == "ClassificationNode":
            self._add_classification_params(node)
        elif cls_name == "OcrNode":
            self._add_ocr_params(node)
        elif cls_name == "RegionOutputNode":
            self._add_region_output_params(node)
        elif cls_name == "ImageOutputNode":
            self._add_output_params(node)
        elif cls_name == "VideoOutputNode":
            self._add_video_output_params(node)
        elif cls_name == "CoordinateTransformNode":
            self._add_coordinate_transform_params(node)

    # ── 参数控件辅助方法 ───────────────────────────────

    def _add_spin(self, label: str, attr: str, node, rng: tuple, step=1):
        spin = QSpinBox()
        spin.setRange(*rng)
        spin.setSingleStep(step)
        spin.setValue(getattr(node, attr))
        spin.setStyleSheet(self._input_style())
        spin.valueChanged.connect(lambda v: setattr(node, attr, v))
        self._form_layout.addRow(label, spin)

    def _add_double_spin(self, label: str, attr: str, node, rng: tuple, step=0.1):
        spin = QDoubleSpinBox()
        spin.setRange(*rng)
        spin.setSingleStep(step)
        spin.setValue(getattr(node, attr))
        spin.setStyleSheet(self._input_style())
        spin.valueChanged.connect(lambda v: setattr(node, attr, v))
        self._form_layout.addRow(label, spin)

    def _add_combo(self, label: str, attr: str, node, items: list):
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentText(getattr(node, attr))
        combo.setStyleSheet(self._input_style())
        combo.currentTextChanged.connect(lambda v: setattr(node, attr, v))
        self._form_layout.addRow(label, combo)

    def _add_check(self, label: str, attr: str, node):
        check = QCheckBox(label)
        check.setChecked(getattr(node, attr))
        check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        check.toggled.connect(lambda v: setattr(node, attr, v))
        self._form_layout.addRow("", check)

    # ── ImageInput ─────────────────────────────────────

    def _add_image_input_params(self, node):
        # ── 模式选择 ─────────────────────────────────
        mode_group = QButtonGroup(self)
        single_rb = QRadioButton("单文件")
        folder_rb = QRadioButton("文件夹")
        mode_group.addButton(single_rb, 0)
        mode_group.addButton(folder_rb, 1)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(single_rb)
        mode_layout.addWidget(folder_rb)
        mode_layout.addStretch()
        self._form_layout.addRow("输入模式:", mode_layout)

        if node.input_mode == "folder":
            folder_rb.setChecked(True)
        else:
            single_rb.setChecked(True)

        # ── 单文件路径 ──────────────────────────────
        self._single_file_widgets: list[QWidget] = []

        single_row = QWidget()
        single_layout = QHBoxLayout(single_row)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(4)
        single_edit = QLineEdit()
        single_edit.setText(node.file_path)
        single_edit.setPlaceholderText("选择图像文件...")
        single_edit.setStyleSheet(self._input_style())
        single_edit.textChanged.connect(lambda v: setattr(node, "file_path", v))
        single_layout.addWidget(single_edit)
        single_btn = QPushButton("...")
        single_btn.setFixedWidth(32)
        single_btn.setStyleSheet(self._btn_style())
        single_btn.clicked.connect(lambda: self._browse_image(node, single_edit))
        single_layout.addWidget(single_btn)
        file_label = QLabel("文件路径:")
        self._form_layout.addRow(file_label, single_row)
        self._single_file_widgets.extend([file_label, single_row])

        # ── 文件夹路径 ──────────────────────────────
        self._folder_widgets: list[QWidget] = []

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(4)
        folder_edit = QLineEdit()
        folder_edit.setText(node.folder_path)
        folder_edit.setPlaceholderText("选择图像文件夹...")
        folder_edit.setStyleSheet(self._input_style())
        folder_edit.textChanged.connect(lambda v: setattr(node, "folder_path", v))
        folder_layout.addWidget(folder_edit)
        folder_btn = QPushButton("...")
        folder_btn.setFixedWidth(32)
        folder_btn.setStyleSheet(self._btn_style())
        folder_btn.clicked.connect(lambda: self._browse_folder(node, folder_edit))
        folder_layout.addWidget(folder_btn)
        folder_label = QLabel("文件夹路径:")
        self._form_layout.addRow(folder_label, folder_row)
        self._folder_widgets.extend([folder_label, folder_row])

        # ── 递归复选框 ──────────────────────────────
        recursive_check = QCheckBox("包含子文件夹")
        recursive_check.setChecked(node.recursive)
        recursive_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        recursive_check.toggled.connect(lambda v: setattr(node, "recursive", v))
        self._form_layout.addRow("", recursive_check)
        self._folder_widgets.append(recursive_check)

        # ── 队列大小 ────────────────────────────────
        queue_spin = QSpinBox()
        queue_spin.setRange(1, 50)
        queue_spin.setValue(node.queue_size)
        queue_spin.setStyleSheet(self._input_style())
        queue_spin.valueChanged.connect(lambda v: setattr(node, "queue_size", v))
        queue_label = QLabel("队列大小:")
        self._form_layout.addRow(queue_label, queue_spin)
        self._folder_widgets.extend([queue_label, queue_spin])

        # ── 可见性切换 ──────────────────────────────
        def _on_mode_toggled():
            is_folder = folder_rb.isChecked()
            node.input_mode = "folder" if is_folder else "single"
            for w in self._single_file_widgets:
                w.setVisible(not is_folder)
            for w in self._folder_widgets:
                w.setVisible(is_folder)

        mode_group.buttonClicked.connect(lambda btn: _on_mode_toggled())
        _on_mode_toggled()  # 初始状态

    # ── VideoInput ────────────────────────────────────

    def _add_video_input_params(self, node):
        # ── 模式选择 ─────────────────────────────────
        mode_group = QButtonGroup(self)
        single_rb = QRadioButton("单文件")
        folder_rb = QRadioButton("文件夹")
        mode_group.addButton(single_rb, 0)
        mode_group.addButton(folder_rb, 1)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(single_rb)
        mode_layout.addWidget(folder_rb)
        mode_layout.addStretch()
        self._form_layout.addRow("输入模式:", mode_layout)

        if node.input_mode == "folder":
            folder_rb.setChecked(True)
        else:
            single_rb.setChecked(True)

        # ── 单文件路径 ──────────────────────────────
        self._video_single_widgets: list[QWidget] = []

        single_row = QWidget()
        single_layout = QHBoxLayout(single_row)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(4)
        single_edit = QLineEdit()
        single_edit.setText(node.file_path)
        single_edit.setPlaceholderText("选择视频文件...")
        single_edit.setStyleSheet(self._input_style())
        single_edit.textChanged.connect(lambda v: setattr(node, "file_path", v))
        single_layout.addWidget(single_edit)
        single_btn = QPushButton("...")
        single_btn.setFixedWidth(32)
        single_btn.setStyleSheet(self._btn_style())
        single_btn.clicked.connect(
            lambda: self._browse_video_file(node, single_edit))
        single_layout.addWidget(single_btn)
        file_label = QLabel("文件路径:")
        self._form_layout.addRow(file_label, single_row)
        self._video_single_widgets.extend([file_label, single_row])

        # ── 文件夹路径 ──────────────────────────────
        self._video_folder_widgets: list[QWidget] = []

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(4)
        folder_edit = QLineEdit()
        folder_edit.setText(node.folder_path)
        folder_edit.setPlaceholderText("选择视频文件夹...")
        folder_edit.setStyleSheet(self._input_style())
        folder_edit.textChanged.connect(lambda v: setattr(node, "folder_path", v))
        folder_layout.addWidget(folder_edit)
        folder_btn = QPushButton("...")
        folder_btn.setFixedWidth(32)
        folder_btn.setStyleSheet(self._btn_style())
        folder_btn.clicked.connect(
            lambda: self._browse_folder(node, folder_edit))
        folder_layout.addWidget(folder_btn)
        folder_label = QLabel("文件夹路径:")
        self._form_layout.addRow(folder_label, folder_row)
        self._video_folder_widgets.extend([folder_label, folder_row])

        # 递归
        recursive_check = QCheckBox("包含子文件夹")
        recursive_check.setChecked(node.recursive)
        recursive_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        recursive_check.toggled.connect(lambda v: setattr(node, "recursive", v))
        self._form_layout.addRow("", recursive_check)
        self._video_folder_widgets.append(recursive_check)

        # ── 队列大小与帧跳过（始终可见）────────────────
        queue_spin = QSpinBox()
        queue_spin.setRange(1, 50)
        queue_spin.setValue(node.queue_size)
        queue_spin.setStyleSheet(self._input_style())
        queue_spin.valueChanged.connect(lambda v: setattr(node, "queue_size", v))
        self._form_layout.addRow("队列大小:", queue_spin)
        skip_spin = QSpinBox()
        skip_spin.setRange(0, 9999)
        skip_spin.setValue(node.frame_skip)
        skip_spin.setStyleSheet(self._input_style())
        skip_spin.valueChanged.connect(lambda v: setattr(node, "frame_skip", v))
        self._form_layout.addRow("帧跳过:", skip_spin)

        # ── 可见性切换 ──────────────────────────────
        def _on_mode_toggled():
            is_folder = folder_rb.isChecked()
            node.input_mode = "folder" if is_folder else "single"
            for w in self._video_single_widgets:
                w.setVisible(not is_folder)
            for w in self._video_folder_widgets:
                w.setVisible(is_folder)

        mode_group.buttonClicked.connect(lambda btn: _on_mode_toggled())
        _on_mode_toggled()  # 初始状态

    # ── RegionInput ───────────────────────────────────

    def _add_region_input_params(self, node):
        path_layout = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setText(node.file_path)
        path_edit.setPlaceholderText("选择图像或视频文件...")
        path_edit.setStyleSheet(self._input_style())
        path_edit.textChanged.connect(lambda v: setattr(node, "file_path", v))
        path_layout.addWidget(path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setStyleSheet(self._btn_style())
        browse_btn.clicked.connect(lambda: self._browse_region_file(node, path_edit))
        path_layout.addWidget(browse_btn)

        self._form_layout.addRow("文件路径:", path_layout)

        draw_btn = QPushButton("✎ 绘制区域")
        draw_btn.setStyleSheet(
            f"QPushButton {{ background: #5cb878; color: #fff; border: none; "
            f"border-radius: 4px; padding: 8px 16px; font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #6dd888; }}"
        )
        draw_btn.clicked.connect(lambda: self._open_region_drawer(node))
        self._form_layout.addRow("", draw_btn)

        self._region_count_label = QLabel()
        self._update_region_count_label(node)
        self._form_layout.addRow("", self._region_count_label)

        # 保存 label 引用用于刷新（在 set_node 中更新）
        self._region_count_label_ref = self._region_count_label

    def _open_region_drawer(self, node):
        node.open_draw_dialog()
        self._update_region_count_label(node)

    def _update_region_count_label(self, node):
        count = getattr(node, "region_count", 0)
        text = f"已绘制 {count} 个区域" if count > 0 else "尚未绘制区域"
        color = "#5cb878" if count > 0 else "#9a9aa0"
        self._region_count_label.setText(text)
        self._region_count_label.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 4px 0; background: transparent;"
        )

    # ── ClassMapping ──────────────────────────────────

    def _add_class_mapping_params(self, node):
        path_layout = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setText(node.json_path)
        path_edit.setPlaceholderText("选择类别映射 JSON 文件...")
        path_edit.setStyleSheet(self._input_style())
        path_edit.textChanged.connect(lambda v: setattr(node, "json_path", v))
        path_layout.addWidget(path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setStyleSheet(self._btn_style())
        browse_btn.clicked.connect(
            lambda: self._browse_json_file(node, path_edit)
        )
        path_layout.addWidget(browse_btn)
        self._form_layout.addRow("映射文件:", path_layout)

        self._mapping_label = QLabel()
        self._refresh_mapping_label(node)
        self._form_layout.addRow("", self._mapping_label)

    def _browse_json_file(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择类别映射 JSON 文件", "",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.json_path = path
            self._refresh_mapping_label(node)

    def _refresh_mapping_label(self, node):
        mapping = getattr(node, "_mapping", {})
        if mapping:
            items = list(mapping.items())[:6]
            text = "\n".join(f"  {k} → {v}" for k, v in items)
            if len(mapping) > 6:
                text += f"\n  ... 共 {len(mapping)} 类"
            color = "#5cb878"
        else:
            text = "尚未加载映射文件"
            color = "#9a9aa0"
        self._mapping_label.setText(text)
        self._mapping_label.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 4px 0; background: transparent;"
        )

    # ── Overlay ───────────────────────────────────────

    def _add_overlay_params(self, node):
        algo = node.algorithm
        if algo == "轮廓绘制":
            self._add_spin("线宽:", "thickness", node, (1, 20))
        elif algo == "半透明填充":
            self._add_double_spin("透明度 (alpha):", "alpha",
                                  node, (0.1, 0.9), step=0.05)

    # ── ObjectDetection ────────────────────────────────

    def _add_object_detection_params(self, node):
        # 模型文件
        path_layout = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setText(node.model_path)
        path_edit.setPlaceholderText("选择 ONNX 模型文件...")
        path_edit.setStyleSheet(self._input_style())
        path_edit.textChanged.connect(lambda v: setattr(node, "model_path", v))
        path_layout.addWidget(path_edit)
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setStyleSheet(self._btn_style())
        browse_btn.clicked.connect(lambda: self._browse_onnx(node, path_edit))
        path_layout.addWidget(browse_btn)
        self._form_layout.addRow("模型文件:", path_layout)

        w_spin = QSpinBox()
        w_spin.setRange(32, 4096)
        w_spin.setValue(node.input_width)
        w_spin.setStyleSheet(self._input_style())
        w_spin.valueChanged.connect(lambda v: setattr(node, "input_width", v))
        self._form_layout.addRow("输入宽度 (W):", w_spin)

        h_spin = QSpinBox()
        h_spin.setRange(32, 4096)
        h_spin.setValue(node.input_height)
        h_spin.setStyleSheet(self._input_style())
        h_spin.valueChanged.connect(lambda v: setattr(node, "input_height", v))
        self._form_layout.addRow("输入高度 (H):", h_spin)

        # 置信度阈值
        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.05, 1.0)
        conf_spin.setSingleStep(0.05)
        conf_spin.setValue(node.conf_threshold)
        conf_spin.setStyleSheet(self._input_style())
        conf_spin.valueChanged.connect(lambda v: setattr(node, "conf_threshold", v))
        self._form_layout.addRow("置信度阈值:", conf_spin)

        # IoU 阈值
        iou_spin = QDoubleSpinBox()
        iou_spin.setRange(0.1, 1.0)
        iou_spin.setSingleStep(0.05)
        iou_spin.setValue(node.iou_threshold)
        iou_spin.setStyleSheet(self._input_style())
        iou_spin.valueChanged.connect(lambda v: setattr(node, "iou_threshold", v))
        self._form_layout.addRow("IoU 阈值 (NMS):", iou_spin)

        # 类别数量
        cls_spin = QSpinBox()
        cls_spin.setRange(1, 1000)
        cls_spin.setValue(node.num_classes)
        cls_spin.setStyleSheet(self._input_style())
        cls_spin.valueChanged.connect(lambda v: setattr(node, "num_classes", v))
        self._form_layout.addRow("类别数量:", cls_spin)

        # 输出框类型
        self._add_combo("输出框类型:", "box_type", node, ["矩形", "圆形"])

        # 推理设备
        self._add_combo("推理设备:", "device", node, ["GPU", "CPU"])

    def _browse_onnx(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ONNX 模型文件", "",
            "ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.model_path = path

    # ── Classification ─────────────────────────────────

    def _add_classification_params(self, node):
        # 模型文件
        path_layout = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setText(node.model_path)
        path_edit.setPlaceholderText("选择 ONNX 模型文件...")
        path_edit.setStyleSheet(self._input_style())
        path_edit.textChanged.connect(lambda v: setattr(node, "model_path", v))
        path_layout.addWidget(path_edit)
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setStyleSheet(self._btn_style())
        browse_btn.clicked.connect(lambda: self._browse_onnx(node, path_edit))
        path_layout.addWidget(browse_btn)
        self._form_layout.addRow("模型文件:", path_layout)

        w_spin = QSpinBox()
        w_spin.setRange(32, 4096)
        w_spin.setValue(node.input_width)
        w_spin.setStyleSheet(self._input_style())
        w_spin.valueChanged.connect(lambda v: setattr(node, "input_width", v))
        self._form_layout.addRow("输入宽度 (W):", w_spin)

        h_spin = QSpinBox()
        h_spin.setRange(32, 4096)
        h_spin.setValue(node.input_height)
        h_spin.setStyleSheet(self._input_style())
        h_spin.valueChanged.connect(lambda v: setattr(node, "input_height", v))
        self._form_layout.addRow("输入高度 (H):", h_spin)

        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.0, 1.0)
        conf_spin.setSingleStep(0.05)
        conf_spin.setDecimals(2)
        conf_spin.setValue(node.conf_threshold)
        conf_spin.setStyleSheet(self._input_style())
        conf_spin.valueChanged.connect(lambda v: setattr(node, "conf_threshold", v))
        self._form_layout.addRow("置信度阈值:", conf_spin)

        cls_spin = QSpinBox()
        cls_spin.setRange(1, 10000)
        cls_spin.setValue(node.num_classes)
        cls_spin.setStyleSheet(self._input_style())
        cls_spin.valueChanged.connect(lambda v: setattr(node, "num_classes", v))
        self._form_layout.addRow("类别数量:", cls_spin)

        topk_spin = QSpinBox()
        topk_spin.setRange(1, 20)
        topk_spin.setValue(node.top_k)
        topk_spin.setStyleSheet(self._input_style())
        topk_spin.valueChanged.connect(lambda v: setattr(node, "top_k", v))
        self._form_layout.addRow("显示 Top-K:", topk_spin)

        self._add_combo("推理设备:", "device", node, ["GPU", "CPU"])

    # ── OCR ──────────────────────────────────────────

    def _add_ocr_params(self, node):
        # 检测模型
        det_path_layout = QHBoxLayout()
        det_edit = QLineEdit()
        det_edit.setText(node.det_model_path)
        det_edit.setPlaceholderText("选择检测 ONNX 模型 (留空=仅识别全图)...")
        det_edit.setStyleSheet(self._input_style())
        det_edit.textChanged.connect(lambda v: setattr(node, "det_model_path", v))
        det_path_layout.addWidget(det_edit)
        det_browse = QPushButton("...")
        det_browse.setFixedWidth(32)
        det_browse.setStyleSheet(self._btn_style())
        det_browse.clicked.connect(lambda: self._browse_onnx(node, det_edit))
        det_path_layout.addWidget(det_browse)
        self._form_layout.addRow("检测模型 (可选):", det_path_layout)

        # 识别模型
        rec_path_layout = QHBoxLayout()
        rec_edit = QLineEdit()
        rec_edit.setText(node.rec_model_path)
        rec_edit.setPlaceholderText("选择识别 ONNX 模型...")
        rec_edit.setStyleSheet(self._input_style())
        rec_edit.textChanged.connect(lambda v: setattr(node, "rec_model_path", v))
        rec_path_layout.addWidget(rec_edit)
        rec_browse = QPushButton("...")
        rec_browse.setFixedWidth(32)
        rec_browse.setStyleSheet(self._btn_style())
        rec_browse.clicked.connect(lambda: self._browse_onnx(node, rec_edit))
        rec_path_layout.addWidget(rec_browse)
        self._form_layout.addRow("识别模型:", rec_path_layout)

        # 字典文件
        dict_path_layout = QHBoxLayout()
        dict_edit = QLineEdit()
        dict_edit.setText(node.dict_path)
        dict_edit.setPlaceholderText("选择字典文件 (.txt)...")
        dict_edit.setStyleSheet(self._input_style())
        dict_edit.textChanged.connect(lambda v: setattr(node, "dict_path", v))
        dict_path_layout.addWidget(dict_edit)
        dict_browse = QPushButton("...")
        dict_browse.setFixedWidth(32)
        dict_browse.setStyleSheet(self._btn_style())
        dict_browse.clicked.connect(
            lambda: self._browse_dict(node, dict_edit)
        )
        dict_path_layout.addWidget(dict_browse)
        self._form_layout.addRow("字典文件:", dict_path_layout)

        # 检测阈值
        det_thresh = QDoubleSpinBox()
        det_thresh.setRange(0.05, 1.0)
        det_thresh.setSingleStep(0.05)
        det_thresh.setDecimals(2)
        det_thresh.setValue(node.det_threshold)
        det_thresh.setStyleSheet(self._input_style())
        det_thresh.valueChanged.connect(lambda v: setattr(node, "det_threshold", v))
        self._form_layout.addRow("检测阈值:", det_thresh)

        # 框置信度
        box_thresh = QDoubleSpinBox()
        box_thresh.setRange(0.05, 1.0)
        box_thresh.setSingleStep(0.05)
        box_thresh.setDecimals(2)
        box_thresh.setValue(node.det_box_thresh)
        box_thresh.setStyleSheet(self._input_style())
        box_thresh.valueChanged.connect(lambda v: setattr(node, "det_box_thresh", v))
        self._form_layout.addRow("框置信度:", box_thresh)

        # 识别阈值
        rec_thresh = QDoubleSpinBox()
        rec_thresh.setRange(0.0, 1.0)
        rec_thresh.setSingleStep(0.05)
        rec_thresh.setDecimals(2)
        rec_thresh.setValue(node.rec_conf_threshold)
        rec_thresh.setStyleSheet(self._input_style())
        rec_thresh.valueChanged.connect(lambda v: setattr(node, "rec_conf_threshold", v))
        self._form_layout.addRow("识别置信度:", rec_thresh)

        self._add_combo("推理设备:", "device", node, ["GPU", "CPU"])

    def _browse_dict(self, node, dict_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择字典文件", "",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if path:
            dict_edit.setText(path)
            node.dict_path = path

    # ── RegionOutput ───────────────────────────────────

    def _add_region_output_params(self, node):
        # ── 批量保存 JSON ────────────────────────────
        save_json_check = QCheckBox("自动保存JSON")
        save_json_check.setChecked(node.save_json)
        save_json_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        self._form_layout.addRow("", save_json_check)

        # 保存目录行（初始根据 save_json 状态显示/隐藏）
        save_dir_row = QWidget()
        save_dir_layout = QHBoxLayout(save_dir_row)
        save_dir_layout.setContentsMargins(0, 0, 0, 0)
        save_dir_layout.setSpacing(4)
        save_dir_edit = QLineEdit()
        save_dir_edit.setText(node.save_dir)
        save_dir_edit.setPlaceholderText("选择批量保存目录...")
        save_dir_edit.setStyleSheet(self._input_style())
        save_dir_edit.textChanged.connect(lambda v: setattr(node, "save_dir", v))
        save_dir_layout.addWidget(save_dir_edit)
        dir_browse_btn = QPushButton("...")
        dir_browse_btn.setFixedWidth(28)
        dir_browse_btn.setStyleSheet(self._btn_style())
        dir_browse_btn.clicked.connect(
            lambda: self._browse_save_dir(node, save_dir_edit)
        )
        save_dir_layout.addWidget(dir_browse_btn)
        save_dir_row.setVisible(node.save_json)
        self._form_layout.addRow("", save_dir_row)

        # 路径为空时的提示标签
        save_dir_hint = QLabel("⚠ 请选择 JSON 保存路径")
        save_dir_hint.setStyleSheet(
            f"color: #e0a050; font-size: 10px; background: transparent;"
        )
        save_dir_hint.setVisible(False)
        self._form_layout.addRow("", save_dir_hint)

        # 勾选逻辑：选中时若路径为空则弹出目录选择，取消则隐藏
        def on_save_json_toggled(checked):
            node.save_json = checked
            save_dir_row.setVisible(checked)
            if checked and not node.save_dir:
                self._browse_save_dir(node, save_dir_edit)
                if not node.save_dir:
                    save_dir_hint.setVisible(True)
                else:
                    save_dir_hint.setVisible(False)
            else:
                save_dir_hint.setVisible(False)

        save_json_check.toggled.connect(on_save_json_toggled)

        # ── 摘要标签 ─────────────────────────────────
        summary_lbl = QLabel(node.region_summary)
        summary_lbl.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; font-size: 12px; "
            f"background: #1e1e24; border-radius: 4px; padding: 8px;"
        )
        summary_lbl.setWordWrap(True)
        self._form_layout.addRow("", summary_lbl)
        self._region_summary_lbl = summary_lbl

        # JSON 预览（截取前几行）
        json_lbl = QLabel()
        json_lbl.setFont(QFont("Consolas", 9))
        json_lbl.setStyleSheet(
            f"color: {SIDEBAR_DIM}; font-size: 10px; "
            f"background: #15151a; border: 1px solid #333; "
            f"border-radius: 4px; padding: 6px;"
        )
        json_lbl.setWordWrap(True)
        if node._last_json:
            preview = node._last_json[:800] + ("..." if len(node._last_json) > 800 else "")
            json_lbl.setText(preview)
        else:
            json_lbl.setText("暂无数据\n请先执行流程")
        self._form_layout.addRow("", json_lbl)
        self._region_json_lbl = json_lbl

        # 按钮
        detail_btn = QPushButton("详细显示")
        detail_btn.setStyleSheet(self._btn_style())
        detail_btn.clicked.connect(lambda: self._on_region_detail(node))
        self._form_layout.addRow("", detail_btn)

        save_btn = QPushButton("下载 JSON")
        save_btn.setStyleSheet(self._btn_style())
        save_btn.clicked.connect(lambda: self._on_region_download(node))
        self._form_layout.addRow("", save_btn)

    def _refresh_region_output(self, node):
        """执行后刷新区域输出面板显示。"""
        if getattr(self, "_region_summary_lbl", None):
            self._region_summary_lbl.setText(node.region_summary)
        if getattr(self, "_region_json_lbl", None) and node._last_json:
            text = node._last_json[:800]
            if len(node._last_json) > 800:
                text += "..."
            self._region_json_lbl.setText(text)

    def _on_region_detail(self, node):
        node.show_detail()

    def _on_region_download(self, node):
        node.save_json()

    # ── Smoothing ──────────────────────────────────────

    def _add_smoothing_params(self, node):
        algo = node.algorithm

        if algo in ("高斯模糊", "均值滤波", "中值滤波"):
            self._add_spin("核大小:", "kernel_size", node, (1, 51), step=2)
        elif algo == "双边滤波":
            self._add_spin("核大小:", "kernel_size", node, (1, 51), step=2)
            self._add_double_spin("颜色标准差 (sigma_color):", "sigma_color",
                                  node, (1.0, 300.0))
            self._add_double_spin("空间标准差 (sigma_space):", "sigma_space",
                                  node, (1.0, 300.0))
        elif algo == "引导滤波":
            self._add_spin("窗口半径:", "radius", node, (1, 60))
            self._add_double_spin("正则化 (epsilon):", "epsilon",
                                  node, (0.0001, 1.0), step=0.001)

    # ── EdgeDetect ─────────────────────────────────────

    def _add_edge_params(self, node):
        algo = node.algorithm

        if algo in ("Sobel", "Laplacian"):
            self._add_spin("核大小:", "kernel_size", node, (1, 31), step=2)
        elif algo == "Canny":
            self._add_spin("核大小 (aperture):", "kernel_size", node, (3, 7), step=2)
            self._add_spin("低阈值:", "threshold_low", node, (0, 500))
            self._add_spin("高阈值:", "threshold_high", node, (0, 500))
            self._add_double_spin("高斯 Sigma:", "sigma", node, (0.0, 10.0))
        elif algo == "Roberts":
            pass  # Roberts 使用固定 2x2 核，无需参数

    # ── Threshold ──────────────────────────────────────

    def _add_threshold_params(self, node):
        algo = node.algorithm

        if algo == "Binary阈值":
            self._add_spin("阈值:", "threshold_value", node, (0, 255))
            self._add_spin("最大值:", "max_value", node, (0, 255))
        elif algo == "Otsu阈值":
            self._add_spin("最大值:", "max_value", node, (0, 255))
        elif algo == "自适应阈值":
            self._add_spin("最大值:", "max_value", node, (0, 255))
            self._add_spin("邻域块大小:", "block_size", node, (3, 99), step=2)
            self._add_spin("常数偏移 C:", "C", node, (-50, 50))

    # ── GeometryTransform ──────────────────────────────

    def _add_geometry_params(self, node):
        algo = node.algorithm

        if algo == "仿射变换":
            self._add_double_spin("旋转角度:", "affine_angle",
                                  node, (0.0, 360.0))
            self._add_double_spin("X 缩放:", "affine_scale_x",
                                  node, (0.1, 5.0), step=0.1)
            self._add_double_spin("Y 缩放:", "affine_scale_y",
                                  node, (0.1, 5.0), step=0.1)
            self._add_spin("X 平移:", "affine_tx", node, (-500, 500))
            self._add_spin("Y 平移:", "affine_ty", node, (-500, 500))

        elif algo == "透视变换":
            self._add_spin("左上 X:", "persp_tl_x", node, (-200, 200))
            self._add_spin("左上 Y:", "persp_tl_y", node, (-200, 200))
            self._add_spin("右上 X:", "persp_tr_x", node, (-200, 200))
            self._add_spin("右上 Y:", "persp_tr_y", node, (-200, 200))
            self._add_spin("左下 X:", "persp_bl_x", node, (-200, 200))
            self._add_spin("左下 Y:", "persp_bl_y", node, (-200, 200))
            self._add_spin("右下 X:", "persp_br_x", node, (-200, 200))
            self._add_spin("右下 Y:", "persp_br_y", node, (-200, 200))

        elif algo == "图像旋转":
            self._add_double_spin("旋转角度:", "rotate_angle",
                                  node, (0.0, 360.0))
            self._add_double_spin("缩放系数:", "rotate_scale",
                                  node, (0.1, 5.0), step=0.1)

        elif algo == "图像缩放":
            self._add_spin("宽度:", "scale_width", node, (1, 4096))
            self._add_spin("高度:", "scale_height", node, (1, 4096))
            self._add_check("保持宽高比", "keep_aspect", node)
            self._add_combo("插值方法:", "interpolation", node,
                            ["最近邻", "双线性", "双三次", "Lanczos"])

    # ── Morphology ─────────────────────────────────────

    def _add_morphology_params(self, node):
        self._add_spin("核大小:", "kernel_size", node, (3, 31), step=2)
        self._add_combo("核形状:", "kernel_shape", node, ["矩形", "椭圆", "十字"])
        self._add_spin("迭代次数:", "iterations", node, (1, 10))

    # ── Enhancement ────────────────────────────────────

    def _add_enhancement_params(self, node):
        algo = node.algorithm

        if algo == "直方图均衡化":
            pass  # 无参数
        elif algo == "对比度拉伸":
            self._add_double_spin("低端裁剪 (%):", "clip_low_pct",
                                  node, (0.0, 49.0), step=0.5)
            self._add_double_spin("高端裁剪 (%):", "clip_high_pct",
                                  node, (0.0, 49.0), step=0.5)
        elif algo == "伽马校正":
            self._add_double_spin("Gamma 值:", "gamma",
                                  node, (0.1, 5.0), step=0.1)
        elif algo == "对数变换":
            self._add_double_spin("缩放系数 C:", "log_c",
                                  node, (0.1, 10.0), step=0.1)
        elif algo == "锐化滤波":
            self._add_double_spin("锐化强度:", "sharpen_amount",
                                  node, (0.1, 5.0), step=0.1)
            self._add_spin("模糊半径:", "sharpen_radius", node, (1, 10))

    # ── Frequency ──────────────────────────────────────

    def _add_frequency_params(self, node):
        algo = node.algorithm

        if algo == "傅里叶变换":
            pass  # 仅显示频谱图，无参数
        elif algo == "高通滤波":
            self._add_spin("截止频率:", "cutoff", node, (1, 200))
        elif algo == "低通滤波":
            self._add_spin("截止频率:", "cutoff", node, (1, 200))
        elif algo == "带通滤波":
            self._add_spin("低截止频率:", "low_cutoff", node, (1, 200))
            self._add_spin("高截止频率:", "high_cutoff", node, (1, 200))
        elif algo == "同态滤波":
            self._add_double_spin("低频增益 (γL):", "homo_low_gamma",
                                  node, (0.1, 2.0), step=0.1)
            self._add_double_spin("高频增益 (γH):", "homo_high_gamma",
                                  node, (0.5, 3.0), step=0.1)
            self._add_spin("截止频率:", "homo_cutoff", node, (1, 200))

    # ── Segmentation ───────────────────────────────────

    def _add_segmentation_params(self, node):
        algo = node.algorithm

        if algo == "区域生长":
            self._add_spin("种子 X:", "seed_x", node, (0, 4096))
            self._add_spin("种子 Y:", "seed_y", node, (0, 4096))
            self._add_spin("生长阈值:", "grow_threshold", node, (1, 100))
        elif algo == "分水岭算法":
            pass  # 自动阈值，无需手动参数
        elif algo == "K-means聚类":
            self._add_spin("聚类数 K:", "k", node, (2, 10))
            self._add_spin("最大迭代:", "kmeans_max_iter", node, (5, 100))
            self._add_spin("尝试次数:", "kmeans_attempts", node, (1, 10))
        elif algo == "GrabCut":
            self._add_spin("迭代次数:", "grabcut_iters", node, (1, 20))
        elif algo == "阈值分割":
            self._add_spin("阈值:", "seg_threshold", node, (0, 255))

    # ── FeatureDetection ─────────────────────────────────

    def _add_feature_detection_params(self, node):
        algo = node.algorithm

        if algo == "Harris角点检测":
            self._add_spin("邻域大小 (blockSize):", "harris_block_size",
                          node, (2, 10))
            self._add_spin("Sobel孔径 (ksize):", "harris_ksize",
                          node, (3, 31), step=2)
            self._add_double_spin("Harris参数 (k):", "harris_k",
                                  node, (0.01, 0.2), step=0.01)
            self._add_double_spin("响应阈值:", "harris_threshold",
                                  node, (0.001, 0.5), step=0.001)
        elif algo == "FAST角点检测":
            self._add_spin("强度阈值:", "fast_threshold", node, (1, 200))
            self._add_check("非极大值抑制", "fast_nonmax", node)
        elif algo == "SIFT特征":
            self._add_spin("最大特征数:", "sift_nfeatures", node, (0, 10000))
            self._add_spin("层数 (nOctaveLayers):", "sift_n_octave_layers",
                          node, (1, 10))
            self._add_double_spin("对比度阈值:", "sift_contrast_threshold",
                                  node, (0.01, 0.2), step=0.01)
            self._add_double_spin("边缘阈值:", "sift_edge_threshold",
                                  node, (1.0, 50.0), step=1.0)
            self._add_double_spin("Sigma:", "sift_sigma",
                                  node, (0.5, 5.0), step=0.1)
        elif algo == "SURF特征":
            self._add_spin("Hessian阈值:", "surf_hessian", node, (100, 10000))
            self._add_spin("八度数 (nOctaves):", "surf_n_octaves", node, (1, 8))
            self._add_spin("层数 (nOctaveLayers):", "surf_n_octave_layers",
                          node, (1, 6))
            self._add_check("扩展描述符 (128维)", "surf_extended", node)
            self._add_check("忽略方向 (Upright)", "surf_upright", node)
        elif algo == "HOG特征":
            self._add_spin("细胞大小:", "hog_cell_size", node, (4, 32), step=4)
            self._add_spin("块大小 (cells):", "hog_block_size", node, (2, 4))
            self._add_spin("方向数 (nbins):", "hog_nbins", node, (4, 18))

    # ── ImageOutput ────────────────────────────────────

    def _add_output_params(self, node):
        # ── 批量保存目录 ────────────────────────────
        save_dir_label = QLabel("保存到:")
        save_dir_label.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; font-size: 11px; background: transparent;"
        )
        self._form_layout.addRow("", save_dir_label)

        save_dir_row = QWidget()
        save_dir_layout = QHBoxLayout(save_dir_row)
        save_dir_layout.setContentsMargins(0, 0, 0, 0)
        save_dir_layout.setSpacing(4)
        save_dir_edit = QLineEdit()
        save_dir_edit.setText(node.save_dir)
        save_dir_edit.setPlaceholderText("选择批量保存目录...")
        save_dir_edit.setStyleSheet(self._input_style())
        save_dir_edit.textChanged.connect(lambda v: setattr(node, "save_dir", v))
        save_dir_layout.addWidget(save_dir_edit)
        dir_browse_btn = QPushButton("...")
        dir_browse_btn.setFixedWidth(28)
        dir_browse_btn.setStyleSheet(self._btn_style())
        dir_browse_btn.clicked.connect(
            lambda: self._browse_save_dir(node, save_dir_edit)
        )
        save_dir_layout.addWidget(dir_browse_btn)
        self._form_layout.addRow("", save_dir_row)

        # 图像预览
        preview = QLabel()
        preview.setFixedSize(184, 140)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet(
            "background: #1a1a20; border: 1px solid #444; "
            "border-radius: 4px; color: #666; font-size: 11px;"
        )
        if node._last_image is not None:
            pixmap = _ndarray_to_qpixmap(node._last_image)
            scaled = pixmap.scaled(
                180, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            preview.setPixmap(scaled)
        else:
            preview.setText("暂无预览\n请先执行流程")
        self._form_layout.addRow("", preview)
        self._preview_label = preview

        self._add_check("执行后自动显示", "_auto_show", node)

        show_btn = QPushButton("显示图像")
        show_btn.setStyleSheet(self._btn_style())
        show_btn.clicked.connect(self._on_output_show)
        self._form_layout.addRow("", show_btn)

        save_btn = QPushButton("保存到文件")
        save_btn.setStyleSheet(self._btn_style())
        save_btn.clicked.connect(self._on_output_save)
        self._form_layout.addRow("", save_btn)

    # ── CoordinateTransform ─────────────────────────────

    def _add_coordinate_transform_params(self, node):
        self._add_spin("目标宽度:", "target_width", node, (1, 65536))
        self._add_spin("目标高度:", "target_height", node, (1, 65536))

    # ── VideoOutput ─────────────────────────────────────

    def _add_video_output_params(self, node):
        # ── 输出模式选择 ─────────────────────────────
        mode_group = QButtonGroup(self)
        video_rb = QRadioButton("视频")
        image_rb = QRadioButton("图片")
        mode_group.addButton(video_rb, 0)
        mode_group.addButton(image_rb, 1)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(video_rb)
        mode_layout.addWidget(image_rb)
        mode_layout.addStretch()
        self._form_layout.addRow("输出模式:", mode_layout)

        if node.output_mode == "image":
            image_rb.setChecked(True)
        else:
            video_rb.setChecked(True)

        # ── 保存目录（始终可见）──────────────────────
        save_dir_row = QWidget()
        save_dir_layout = QHBoxLayout(save_dir_row)
        save_dir_layout.setContentsMargins(0, 0, 0, 0)
        save_dir_layout.setSpacing(4)
        save_dir_edit = QLineEdit()
        save_dir_edit.setText(node.save_dir)
        save_dir_edit.setPlaceholderText("选择保存目录...")
        save_dir_edit.setStyleSheet(self._input_style())
        save_dir_edit.textChanged.connect(lambda v: setattr(node, "save_dir", v))
        save_dir_layout.addWidget(save_dir_edit)
        dir_browse_btn = QPushButton("...")
        dir_browse_btn.setFixedWidth(28)
        dir_browse_btn.setStyleSheet(self._btn_style())
        dir_browse_btn.clicked.connect(
            lambda: self._browse_save_dir(node, save_dir_edit)
        )
        save_dir_layout.addWidget(dir_browse_btn)
        save_dir_label = QLabel("保存目录:")
        self._form_layout.addRow(save_dir_label, save_dir_row)

        # ── 视频模式控件 ─────────────────────────────
        self._video_mode_widgets: list[QWidget] = []

        video_fmt_combo = QComboBox()
        video_fmt_combo.addItems(["mp4", "avi", "mov"])
        video_fmt_combo.setCurrentText(node.video_format)
        video_fmt_combo.setStyleSheet(self._input_style())
        video_fmt_combo.currentTextChanged.connect(
            lambda v: setattr(node, "video_format", v))
        video_fmt_label = QLabel("视频格式:")
        self._form_layout.addRow(video_fmt_label, video_fmt_combo)
        self._video_mode_widgets.extend([video_fmt_label, video_fmt_combo])

        fps_label = QLabel("帧率 (FPS):")
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 120)
        fps_spin.setValue(node.fps)
        fps_spin.setStyleSheet(self._input_style())
        fps_spin.valueChanged.connect(lambda v: setattr(node, "fps", v))
        self._form_layout.addRow(fps_label, fps_spin)
        self._video_mode_widgets.extend([fps_label, fps_spin])

        # ── 图片模式控件 ─────────────────────────────
        self._image_mode_widgets: list[QWidget] = []

        img_fmt_combo = QComboBox()
        img_fmt_combo.addItems(["jpg", "png"])
        img_fmt_combo.setCurrentText(node.image_format)
        img_fmt_combo.setStyleSheet(self._input_style())
        img_fmt_combo.currentTextChanged.connect(
            lambda v: setattr(node, "image_format", v))
        img_fmt_label = QLabel("图像格式:")
        self._form_layout.addRow(img_fmt_label, img_fmt_combo)
        self._image_mode_widgets.extend([img_fmt_label, img_fmt_combo])

        # ── 保存结果复选框 ─────────────────────────────
        auto_save_check = QCheckBox("保存结果")
        auto_save_check.setChecked(node.auto_save)
        auto_save_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        auto_save_check.toggled.connect(lambda v: setattr(node, "auto_save", v))
        self._form_layout.addRow("", auto_save_check)

        # ── 可见性切换 ──────────────────────────────
        def _on_output_mode_toggled():
            is_video = video_rb.isChecked()
            node.output_mode = "video" if is_video else "image"
            for w in self._video_mode_widgets:
                w.setVisible(is_video)
            for w in self._image_mode_widgets:
                w.setVisible(not is_video)

        mode_group.buttonClicked.connect(
            lambda btn: _on_output_mode_toggled())
        _on_output_mode_toggled()  # 初始状态

    # ── 通用 ───────────────────────────────────────────

    def _browse_image(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.file_path = path

    def _browse_video_file(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.file_path = path

    def _browse_folder(self, node, path_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if path:
            path_edit.setText(path)
            node.folder_path = path

    def _browse_save_dir(self, node, path_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if path:
            path_edit.setText(path)
            node.save_dir = path

    def _browse_region_file(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像或视频文件", "",
            "媒体文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp "
            "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;"
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;"
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;"
            "所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.file_path = path

    def _on_output_show(self):
        node = self._current_node
        if node is None or node._last_image is None:
            QMessageBox.information(self, "提示", "请先执行流程生成结果图像。")
            return
        node.show_last_result()

    def _on_output_save(self):
        node = self._current_node
        if node is None or node._last_image is None:
            QMessageBox.information(self, "提示", "请先执行流程生成结果图像。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "output.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;所有文件 (*)"
        )
        if path:
            node.save_last_result(path)

    def _input_style(self):
        return (
            f"background: {BTN_BG}; color: {SIDEBAR_TEXT}; border: 1px solid #555; "
            f"border-radius: 4px; padding: 4px 6px; font-size: 11px;"
        )

    def _btn_style(self):
        return (
            f"QPushButton {{ background: {ACCENT}; color: #fff; border: none; "
            f"border-radius: 4px; padding: 5px 12px; font-size: 11px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {ACCENT_HOVER}; }}"
        )


# ── 主窗口 ────────────────────────────────────────────
class MainWindow(QMainWindow):
    """应用程序主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理流式编辑器 — Image Flow")
        self.resize(1400, 850)
        self.setMinimumSize(1000, 600)

        self.engine = ExecutionEngine()
        self._runner: Optional[FlowRunner] = None
        self._stop_action = None
        self._current_project_path: str = ""
        self._build_ui()
        self._apply_theme()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 从左侧拖拽算子到画布开始构建流程")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #3a3a40; }")
        splitter.setHandleWidth(1)

        # ── 左侧：节点面板 ────────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(210)
        left_panel.setStyleSheet(f"background: {SIDEBAR_BG};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(NodePanel())

        # ── 中间：画布（上） + 日志（下） ──────────────
        center_widget = QWidget()
        center_widget.setStyleSheet("background: #1e1e23;")
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.scene = NodeScene(self.engine)
        self.canvas = NodeCanvas(self.scene)
        self.canvas.setAcceptDrops(True)
        self.canvas.dragEnterEvent = self._canvas_drag_enter
        self.canvas.dragMoveEvent = self._canvas_drag_move
        self.canvas.dropEvent = self._canvas_drop
        self.scene.selectionChanged.connect(self._on_selection_changed)
        center_layout.addWidget(self.canvas, stretch=1)

        self.log_panel = LogPanel()
        self.log_panel.setFixedHeight(150)
        self.log_panel.setStyleSheet("background: #1a1a1f;")
        center_layout.addWidget(self.log_panel)

        # ── 右侧：属性面板 ────────────────────────────
        right_panel = QWidget()
        right_panel.setFixedWidth(240)
        right_panel.setStyleSheet(f"background: {SIDEBAR_BG};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.property_panel = PropertyPanel()
        right_layout.addWidget(self.property_panel)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_panel)
        splitter.setSizes([210, 950, 240])
        main_layout.addWidget(splitter)

        # 菜单栏 / 工具栏
        self._build_menubar()
        self._build_toolbar()

    def _build_menubar(self):
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
            f"QMenuBar {{ background: #1e1e24; color: {SIDEBAR_TEXT}; padding: 2px; "
            f"border-bottom: 1px solid #333; }}"
            f"QMenuBar::item:selected {{ background: {BTN_HOVER}; }}"
            f"QMenu {{ background: #2b2b30; color: {SIDEBAR_TEXT}; border: 1px solid #444; }}"
            f"QMenu::item:selected {{ background: {ACCENT}; }}"
        )

        file_menu = menu_bar.addMenu("文件(&F)")
        file_menu.addAction("新建工程(&N)", self._new_project, Qt.Key_N | Qt.CTRL)
        file_menu.addAction("读取工程(&O)", self._open_project, Qt.Key_O | Qt.CTRL)
        file_menu.addAction("保存工程(&S)", self._save_project, Qt.Key_S | Qt.CTRL)
        file_menu.addSeparator()
        file_menu.addAction("清空画布", self._clear_canvas)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, Qt.Key_Q | Qt.CTRL)

        run_menu = menu_bar.addMenu("运行(&R)")
        run_menu.addAction("执行流程(&E)", self._execute_flow, Qt.Key_F5)

        help_menu = menu_bar.addMenu("帮助(&H)")
        help_menu.addAction("使用说明", self._show_help)

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setStyleSheet(
            f"QToolBar {{ background: #1e1e24; border-bottom: 1px solid #333; "
            f"padding: 4px 8px; spacing: 6px; }}"
            f"QToolButton {{ color: {SIDEBAR_TEXT}; padding: 4px 10px; "
            f"border-radius: 4px; font-size: 12px; }}"
            f"QToolButton:hover {{ background: {BTN_HOVER}; }}"
        )

        run_action = QAction("▶ 执行", self)
        run_action.setToolTip("执行当前节点流程 (F5)")
        run_action.triggered.connect(self._execute_flow)
        toolbar.addAction(run_action)

        self._stop_action = QAction("■ 中止", self)
        self._stop_action.setToolTip("中止正在执行的流程")
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._cancel_flow)
        toolbar.addAction(self._stop_action)

        toolbar.addSeparator()

        save_action = QAction("💾 保存", self)
        save_action.setToolTip("保存工程到文件 (Ctrl+S)")
        save_action.triggered.connect(self._save_project)
        toolbar.addAction(save_action)

        open_action = QAction("📂 读取", self)
        open_action.setToolTip("读取工程文件 (Ctrl+O)")
        open_action.triggered.connect(self._open_project)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        clear_action = QAction("✕ 清空", self)
        clear_action.setToolTip("清空画布上所有节点和连线")
        clear_action.triggered.connect(self._clear_canvas)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        fit_action = QAction("⊞ 适应画布", self)
        fit_action.setToolTip("缩放并居中显示所有节点")
        fit_action.triggered.connect(self._fit_canvas)
        toolbar.addAction(fit_action)

        self.addToolBar(toolbar)

    def _apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 35))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 225))
        palette.setColor(QPalette.Base, QColor(35, 35, 40))
        palette.setColor(QPalette.Text, QColor(220, 220, 225))
        palette.setColor(QPalette.Button, QColor(60, 60, 70))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 225))
        palette.setColor(QPalette.Highlight, QColor(90, 156, 248))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 130))
        self.setPalette(palette)
        self.setStyleSheet(
            f"QMainWindow {{ background: #1e1e23; }}"
            f"QToolTip {{ background: #333; color: {SIDEBAR_TEXT}; border: 1px solid #555; }}"
        )

    # ── 画布拖拽 ──────────────────────────────────────
    def _canvas_drag_enter(self, event):
        if event.mimeData().hasFormat("application/x-node-type"):
            event.acceptProposedAction()

    def _canvas_drag_move(self, event):
        if event.mimeData().hasFormat("application/x-node-type"):
            event.acceptProposedAction()

    def _canvas_drop(self, event):
        mime = event.mimeData()
        if mime.hasFormat("application/x-node-type"):
            node_name = mime.data("application/x-node-type").data().decode()
            registry = NodeRegistry()
            cls = registry.get(node_name)
            if cls:
                scene_pos = self.canvas.mapToScene(event.pos())
                self.scene.add_node(cls, scene_pos)
                self.status_bar.showMessage(f"已添加: {node_name}")
                self.log_panel.info(f"添加节点: {node_name}")
            event.acceptProposedAction()

    # ── 节点操作 ──────────────────────────────────────
    def add_node_to_center(self, node_cls: Type[Node]):
        """在画布可见区域中心添加节点。"""
        view_center = self.canvas.mapToScene(
            self.canvas.viewport().rect().center()
        )
        offset = QPointF(
            (self.scene.node_items.__len__() % 5) * 30,
            (self.scene.node_items.__len__() % 5) * 30,
        )
        self.scene.add_node(node_cls, view_center + offset)
        self.status_bar.showMessage(f"已添加: {node_cls.display_name}")
        self.log_panel.info(f"添加节点: {node_cls.display_name}")

    def _clear_canvas(self):
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空画布上的所有节点和连线吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._do_clear_canvas()
            self.status_bar.showMessage("画布已清空")
            self.log_panel.warn("画布已清空")

    def _do_clear_canvas(self):
        """清空画布和引擎（不弹出确认框）。"""
        for uid in list(self.scene.node_items.keys()):
            item = self.scene.node_items[uid]
            self.scene.remove_node_item(item)
        self.scene.node_items.clear()
        self.scene.wire_items.clear()
        self.engine.nodes.clear()
        self.engine.connections.clear()
        self.property_panel.set_node(None)

    # ── 工程管理 ──────────────────────────────────────

    def _new_project(self):
        if self.engine.nodes:
            reply = QMessageBox.question(
                self, "新建工程", "当前工程尚未保存，确定要新建工程吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self._do_clear_canvas()
        self._current_project_path = ""
        self.setWindowTitle("图像处理流式编辑器 — Image Flow [未命名]")
        self.status_bar.showMessage("已创建新工程")
        self.log_panel.info("新建工程")

    def _save_project(self):
        if not self._current_project_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "保存工程", "flow.json",
                "Image Flow 工程 (*.json);;所有文件 (*)"
            )
            if not path:
                return
            self._current_project_path = path

        try:
            save_project(self.engine, self.scene, self._current_project_path)
            fname = os.path.basename(self._current_project_path)
            self.setWindowTitle(f"图像处理流式编辑器 — Image Flow [{fname}]")
            self.status_bar.showMessage(f"工程已保存: {self._current_project_path}")
            self.log_panel.success(f"工程已保存: {self._current_project_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            self.log_panel.error(f"保存工程失败: {e}")

    def _open_project(self):
        if self.engine.nodes:
            reply = QMessageBox.question(
                self, "读取工程", "当前工程尚未保存，确定要读取新工程吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        path, _ = QFileDialog.getOpenFileName(
            self, "读取工程", "",
            "Image Flow 工程 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        try:
            data = load_project(path)
            self._do_clear_canvas()
            self._rebuild_scene(data)
            self._current_project_path = path
            fname = os.path.basename(path)
            self.setWindowTitle(f"图像处理流式编辑器 — Image Flow [{fname}]")
            self.status_bar.showMessage(f"工程已加载: {path} — {len(data.get('nodes', []))} 个节点")
            self.log_panel.success(f"工程已加载: {path}")
            self._fit_canvas()
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"无法读取工程文件:\n{e}")
            self.log_panel.error(f"读取工程失败: {e}")

    def _rebuild_scene(self, data: dict):
        """根据工程数据重建场景中的节点和连线。"""
        registry = NodeRegistry()
        cls_map: dict[str, type] = {}
        for cls in registry.list_all().values():
            cls_map[cls.__name__] = cls

        node_map: dict[str, Node] = {}

        for nd in data.get("nodes", []):
            cls_name = nd["type"]
            cls = cls_map.get(cls_name)
            if cls is None:
                self.log_panel.warn(f"跳过未知节点类型: {cls_name}")
                continue

            node = cls()
            node.uid = nd["uid"]
            apply_params(node, nd.get("params", {}))

            pos = QPointF(nd.get("x", 0), nd.get("y", 0))
            item = NodeItem(node)
            item.setPos(pos)
            self.scene.addItem(item)
            self.scene.node_items[node.uid] = item
            item.node_moved.connect(self.scene._on_node_moved)
            for port_item in item.port_items.values():
                port_item.port_clicked.connect(self.scene._on_port_clicked)

            self.engine.add_node(node)
            node_map[node.uid] = node

        for cd in data.get("connections", []):
            src_node = node_map.get(cd["src_node"])
            tgt_node = node_map.get(cd["tgt_node"])
            if not src_node or not tgt_node:
                continue
            src_port = src_node.outputs.get(cd["src_port"])
            tgt_port = tgt_node.inputs.get(cd["tgt_port"])
            if not src_port or not tgt_port:
                continue

            conn = Connection(src_port, tgt_port)
            conn.uid = cd["uid"]
            self.engine.add_connection(conn)

            wire = WireItem(conn)
            self.scene.addItem(wire)
            wire.update_path()
            self.scene.wire_items[conn.uid] = wire

    def _fit_canvas(self):
        if self.scene.node_items:
            self.canvas.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    # ── 选中处理 ──────────────────────────────────────
    def _on_selection_changed(self):
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], NodeItem):
            item = selected[0]
            self.property_panel.set_node(item.node, item)
        else:
            self.property_panel.set_node(None)

    # ── 流程执行 ──────────────────────────────────────
    def _cancel_flow(self):
        """中止正在执行的流程。"""
        if self._runner is not None and self._runner.isRunning():
            self._runner.cancel()
            self.log_panel.warn("正在中止流程 — 等待当前帧处理完成...")
            self.status_bar.showMessage("正在中止流程...")
            self._stop_action.setEnabled(False)

    def _execute_flow(self):
        if not self.engine.nodes:
            QMessageBox.information(self, "提示", "画布中没有节点，请先添加算子。")
            return

        has_input = any(len(n.inputs) == 0 for n in self.engine.nodes)
        if not has_input:
            QMessageBox.warning(self, "警告", "流程中缺少数据源节点（如「图像读取」）。")
            return

        if self._runner is not None and self._runner.isRunning():
            QMessageBox.information(self, "提示", "流程正在执行中，请稍候...")
            return

        self.status_bar.showMessage("正在执行流程...")
        self.log_panel.info(f"开始执行流程 — {len(self.engine.nodes)} 个节点")

        self._stop_action.setEnabled(True)

        # 断开之前的 batch_progress 信号（如果存在）
        if self._runner is not None:
            try:
                self._runner.batch_progress.disconnect()
            except TypeError:
                pass

        self._runner = FlowRunner(self.engine)
        self._runner.finished.connect(self._on_flow_finished)
        self._runner.error.connect(self._on_flow_error)
        self._runner.node_started.connect(
            lambda name: self.log_panel.info(f"正在执行: {name}")
        )
        self._runner.node_done.connect(
            lambda name: self.log_panel.success(f"执行完成: {name}")
        )
        self._runner.node_error.connect(
            lambda name, msg: self.log_panel.error(f"节点错误 [{name}]: {msg}")
        )
        self._runner.batch_progress.connect(self._on_batch_progress)
        self._runner.start()

    def _on_batch_progress(self, current: int, total: int, filename: str):
        self.status_bar.showMessage(
            f"批处理中... [{current}/{total}] {filename}"
        )
        self.log_panel.info(f"  [{current}/{total}] 处理: {filename}")

    def _on_flow_finished(self, results):
        self._runner = None
        self._stop_action.setEnabled(False)

        # ── 批处理结果 ──────────────────────────────
        if isinstance(results, dict) and results.get("mode") == "batch":
            total = results.get("total", 0)
            completed = results.get("completed", 0)
            cancelled = self.engine._cancelled
            if cancelled:
                self.status_bar.showMessage(
                    f"流程已中止 — 共处理 {completed} 帧"
                )
                self.log_panel.warn(f"流程已中止，共处理 {completed} 帧")
            else:
                self.status_bar.showMessage(
                    f"批处理完成: {completed}/{total} 帧 ✓"
                )
                self.log_panel.success(f"算法运行完成，批处理 {completed}/{total} 帧")

            # 报告文件级错误
            errors = results.get("errors", [])
            if errors:
                self.log_panel.warn(f"共 {len(errors)} 个文件读取失败:")
                for idx, filepath, msg in errors:
                    self.log_panel.warn(f"  [{os.path.basename(filepath)}] {msg}")

            self._update_node_statuses()
            self.property_panel.refresh_output_preview()
            self.property_panel._refresh_region_output_from_nodes(self.engine.nodes)
            return

        # ── 单次执行结果 ────────────────────────────
        self.status_bar.showMessage("流程执行完成 ✓")
        node_count = len(self.engine.nodes)
        self.log_panel.success(f"算法运行完成，共处理 {node_count} 个节点")

        # 输出检测/分类结果到日志
        for node in self.engine.nodes:
            summary = getattr(node, "_result_summary", "")
            if summary:
                for line in summary.split("\n"):
                    self.log_panel.info(f"  [{node.display_name}] {line}")

        self._update_node_statuses()
        self.property_panel.refresh_output_preview()
        self.property_panel._refresh_region_output_from_nodes(self.engine.nodes)
        # 刷新类别映射标签
        if self.property_panel._current_node is not None and \
                hasattr(self.property_panel._current_node, "json_path"):
            self.property_panel._refresh_mapping_label(
                self.property_panel._current_node
            )
        for node in self.engine.nodes:
            if hasattr(node, "_auto_show") and node._auto_show and node._last_image is not None:
                node.show_last_result()

    def _on_flow_error(self, err_msg):
        self._runner = None
        self._stop_action.setEnabled(False)
        self.status_bar.showMessage("执行失败")
        self.log_panel.error(f"流程执行失败:\n{err_msg}")
        QMessageBox.critical(
            self, "执行错误",
            f"流程执行过程中发生错误:\n\n{err_msg}"
        )

    def _update_node_statuses(self):
        """执行后更新节点外观。"""
        for item in self.scene.node_items.values():
            item.update()

    # ── 帮助 ──────────────────────────────────────────
    def _show_help(self):
        text = (
            "<h3>图像处理流式编辑器 — 使用说明</h3>"
            "<ol>"
            "<li>从<b>左侧面板</b>拖拽算子到画布上（或双击自动添加）。</li>"
            "<li>点击算子的<b>输出端口</b>（蓝色圆点），拖拽到另一个算子的<b>输入端口</b>（红色圆点）来建立连线。</li>"
            "<li>右键点击节点或连线可以<b>删除</b>。</li>"
            "<li>选中节点后在<b>右侧属性面板</b>中配置参数。</li>"
            "<li>点击工具栏的<b>「执行」</b>按钮（或按 F5）运行流程。</li>"
            "<li>使用<b>鼠标中键</b>拖拽平移画布，<b>滚轮</b>缩放画布。</li>"
            "</ol>"
        )
        QMessageBox.information(self, "使用说明", text)


# ── 入口 ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ImageFlow")

    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
