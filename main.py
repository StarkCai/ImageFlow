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

from PyQt5.QtCore import Qt, QMimeData, QPointF, QSize, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QDrag, QPixmap, QPainter, QPen, QBrush, QIcon,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QLineEdit, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QToolBar, QAction, QStatusBar, QSizePolicy, QCheckBox, QMenuBar,
    QMenu, QTextEdit,
)

# 必须在导入节点之前设置 Qt 平台插件路径
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

from node_base import Node, Port, Connection, ExecutionEngine
from node_registry import NodeRegistry
from node_canvas import NodeScene, NodeCanvas, NodeItem
from nodes.image_output import _ndarray_to_qpixmap

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


# ── 可拖拽的节点列表 ─────────────────────────────────
class NodeListItem(QWidget):
    """节点面板中的每一项，支持拖拽。"""

    def __init__(self, node_cls: Type[Node], parent=None):
        super().__init__(parent)
        self.node_cls = node_cls
        self.setToolTip(f"{node_cls.category} → {node_cls.display_name}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)

        # 类别色标
        color_map = {"输入": ACCENT, "图像处理": SUCCESS, "输出": "#f0a050"}
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
        container_layout.setSpacing(12)

        self._list_items: dict[str, NodeListItem] = {}

        for cat_name, node_classes in grouped.items():
            cat_label = QLabel(cat_name)
            cat_label.setStyleSheet(
                f"color: {SIDEBAR_DIM}; font-size: 10px; font-weight: bold; "
                f"text-transform: uppercase; padding: 2px 4px; background: transparent;"
            )
            container_layout.addWidget(cat_label)

            for cls in node_classes:
                item = NodeListItem(cls)
                item.mouseDoubleClickEvent = lambda e, c=cls: self._on_double_click(c)
                container_layout.addWidget(item)
                self._list_items[cls.display_name] = item

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def _on_double_click(self, node_cls: Type[Node]):
        """双击默认将节点添加到画布中心。"""
        main_win = self.window()
        if isinstance(main_win, MainWindow):
            main_win.add_node_to_center(node_cls)


# ── 属性面板 ──────────────────────────────────────────
class PropertyPanel(QWidget):
    """右侧属性编辑面板，显示选中节点的可配置参数。"""

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
        layout.setSpacing(6)

        title = QLabel("属性面板")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setStyleSheet(f"color: {SIDEBAR_TEXT}; padding: 4px 0; background: transparent;")
        layout.addWidget(title)

        self._hint = QLabel("选择一个节点以编辑参数")
        self._hint.setStyleSheet(f"color: {SIDEBAR_DIM}; font-size: 10px; background: transparent;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 3px; }}"
        )
        self._scroll = scroll

        self._form_container = QWidget()
        self._form_container.setStyleSheet("background: transparent;")
        self._form_layout = QFormLayout(self._form_container)
        self._form_layout.setContentsMargins(4, 4, 4, 4)
        self._form_layout.setSpacing(8)
        self._form_layout.setLabelAlignment(Qt.AlignLeft)

        scroll.setWidget(self._form_container)
        layout.addWidget(scroll)
        layout.addStretch()

    def set_node(self, node: Optional[Node], item: Optional[NodeItem] = None):
        """加载节点属性到面板。"""
        self._current_node = node
        self._current_item = item
        self._clear_form()

        if node is None:
            self._hint.setText("选择一个节点以编辑参数")
            self._hint.setVisible(True)
            return

        self._hint.setVisible(False)

        # 节点标题
        title_label = QLabel(f"{node.display_name}")
        title_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        self._form_layout.addRow(title_label)

        uid_label = QLabel(f"ID: {node.uid}")
        uid_label.setStyleSheet(f"color: {SIDEBAR_DIM}; font-size: 10px; background: transparent;")
        self._form_layout.addRow(uid_label)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: #444;")
        self._form_layout.addRow(sep)

        # 为特定节点类型添加参数编辑器
        self._add_node_params(node)

    def _clear_form(self):
        self._preview_label = None
        # 逐行移除所有表单项
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)
        # 删除残留的子控件（包括嵌套布局中的控件）
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
            208, 158, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        preview.setPixmap(scaled)

    def _add_node_params(self, node: Node):
        """根据节点类型动态创建参数控件。"""
        cls_name = node.__class__.__name__

        if cls_name == "ImageInputNode":
            self._add_image_input_params(node)

        elif cls_name == "GrayscaleNode":
            self._add_grayscale_params(node)

        elif cls_name == "BlurNode":
            self._add_blur_params(node)

        elif cls_name == "EdgeDetectNode":
            self._add_edge_params(node)

        elif cls_name == "ThresholdNode":
            self._add_threshold_params(node)

        elif cls_name == "ResizeNode":
            self._add_resize_params(node)

        elif cls_name == "ImageOutputNode":
            self._add_output_params(node)

    def _add_image_input_params(self, node):
        path_layout = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setText(node.file_path)
        path_edit.setPlaceholderText("选择图像文件...")
        path_edit.setStyleSheet(self._input_style())
        path_edit.textChanged.connect(lambda v: setattr(node, "file_path", v))
        path_layout.addWidget(path_edit)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(32)
        browse_btn.setStyleSheet(self._btn_style())
        browse_btn.clicked.connect(
            lambda: self._browse_image(node, path_edit)
        )
        path_layout.addWidget(browse_btn)

        self._form_layout.addRow("文件路径:", path_layout)

    def _add_grayscale_params(self, node):
        method_combo = QComboBox()
        method_combo.addItems(["luminosity", "average", "lightness"])
        method_combo.setCurrentText(node.method)
        method_combo.setStyleSheet(self._input_style())
        method_combo.currentTextChanged.connect(lambda v: setattr(node, "method", v))
        self._form_layout.addRow("转换方法:", method_combo)

    def _add_blur_params(self, node):
        ksize_spin = QSpinBox()
        ksize_spin.setRange(1, 51)
        ksize_spin.setSingleStep(2)
        ksize_spin.setValue(node.kernel_size)
        ksize_spin.setStyleSheet(self._input_style())
        ksize_spin.valueChanged.connect(lambda v: setattr(node, "kernel_size", v))
        self._form_layout.addRow("核大小:", ksize_spin)

        sigma_spin = QDoubleSpinBox()
        sigma_spin.setRange(0.1, 30.0)
        sigma_spin.setSingleStep(0.5)
        sigma_spin.setValue(node.sigma)
        sigma_spin.setStyleSheet(self._input_style())
        sigma_spin.valueChanged.connect(lambda v: setattr(node, "sigma", v))
        self._form_layout.addRow("Sigma:", sigma_spin)

    def _add_edge_params(self, node):
        t1_spin = QSpinBox()
        t1_spin.setRange(0, 500)
        t1_spin.setValue(node.threshold1)
        t1_spin.setStyleSheet(self._input_style())
        t1_spin.valueChanged.connect(lambda v: setattr(node, "threshold1", v))
        self._form_layout.addRow("低阈值:", t1_spin)

        t2_spin = QSpinBox()
        t2_spin.setRange(0, 500)
        t2_spin.setValue(node.threshold2)
        t2_spin.setStyleSheet(self._input_style())
        t2_spin.valueChanged.connect(lambda v: setattr(node, "threshold2", v))
        self._form_layout.addRow("高阈值:", t2_spin)

    def _add_threshold_params(self, node):
        thresh_spin = QSpinBox()
        thresh_spin.setRange(0, 255)
        thresh_spin.setValue(node.threshold_value)
        thresh_spin.setStyleSheet(self._input_style())
        thresh_spin.valueChanged.connect(lambda v: setattr(node, "threshold_value", v))
        self._form_layout.addRow("阈值:", thresh_spin)

        method_combo = QComboBox()
        method_combo.addItems(["binary", "binary_inv", "trunc", "tozero", "otsu"])
        method_combo.setCurrentText(node.method)
        method_combo.setStyleSheet(self._input_style())
        method_combo.currentTextChanged.connect(lambda v: setattr(node, "method", v))
        self._form_layout.addRow("方法:", method_combo)

    def _add_resize_params(self, node):
        w_spin = QSpinBox()
        w_spin.setRange(1, 4096)
        w_spin.setValue(node.width)
        w_spin.setStyleSheet(self._input_style())
        w_spin.valueChanged.connect(lambda v: setattr(node, "width", v))
        self._form_layout.addRow("宽度:", w_spin)

        h_spin = QSpinBox()
        h_spin.setRange(1, 4096)
        h_spin.setValue(node.height)
        h_spin.setStyleSheet(self._input_style())
        h_spin.valueChanged.connect(lambda v: setattr(node, "height", v))
        self._form_layout.addRow("高度:", h_spin)

        aspect_check = QCheckBox("保持宽高比")
        aspect_check.setChecked(node.keep_aspect)
        aspect_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        aspect_check.toggled.connect(lambda v: setattr(node, "keep_aspect", v))
        self._form_layout.addRow("", aspect_check)

        interp_combo = QComboBox()
        interp_combo.addItems(["最近邻", "双线性", "双三次", "Lanczos"])
        interp_combo.setCurrentText(node.interpolation)
        interp_combo.setStyleSheet(self._input_style())
        interp_combo.currentTextChanged.connect(lambda v: setattr(node, "interpolation", v))
        self._form_layout.addRow("插值方法:", interp_combo)

    def _add_output_params(self, node):
        # 图像预览
        preview = QLabel()
        preview.setFixedSize(210, 160)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet(
            "background: #1a1a20; border: 1px solid #444; "
            "border-radius: 4px; color: #666; font-size: 11px;"
        )
        if node._last_image is not None:
            pixmap = _ndarray_to_qpixmap(node._last_image)
            scaled = pixmap.scaled(
                208, 158, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            preview.setPixmap(scaled)
        else:
            preview.setText("暂无预览图像\n请先执行流程")
        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addStretch()
        wrapper.addWidget(preview)
        wrapper.addStretch()
        self._form_layout.addRow("", wrapper)
        self._preview_label = preview

        auto_check = QCheckBox("执行后自动显示")
        auto_check.setChecked(node._auto_show)
        auto_check.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; background: transparent; spacing: 6px;"
        )
        auto_check.toggled.connect(lambda v: setattr(node, "_auto_show", v))
        self._form_layout.addRow("", auto_check)

        save_btn = QPushButton("保存到文件")
        save_btn.setStyleSheet(self._btn_style())
        save_btn.clicked.connect(self._on_output_save)
        self._form_layout.addRow("", save_btn)

    def _browse_image(self, node, path_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        if path:
            path_edit.setText(path)
            node.file_path = path

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

        # 左侧节点面板
        left_panel = QWidget()
        left_panel.setFixedWidth(210)
        left_panel.setStyleSheet(f"background: {SIDEBAR_BG};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(NodePanel())

        # 中间画布
        self.scene = NodeScene(self.engine)
        self.canvas = NodeCanvas(self.scene)
        self.canvas.setAcceptDrops(True)
        self.canvas.dragEnterEvent = self._canvas_drag_enter
        self.canvas.dragMoveEvent = self._canvas_drag_move
        self.canvas.dropEvent = self._canvas_drop

        # 监听选中变化
        self.scene.selectionChanged.connect(self._on_selection_changed)

        # 右侧属性面板
        right_panel = QWidget()
        right_panel.setFixedWidth(240)
        right_panel.setStyleSheet(f"background: {SIDEBAR_BG};")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.property_panel = PropertyPanel()
        right_layout.addWidget(self.property_panel)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([210, 950, 240])
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #3a3a40; }")
        main_layout.addWidget(splitter)

        # 菜单栏
        self._build_menubar()

        # 工具栏
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

    def _clear_canvas(self):
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空画布上的所有节点和连线吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for uid in list(self.scene.node_items.keys()):
                item = self.scene.node_items[uid]
                self.scene.remove_node_item(item)
            self.property_panel.set_node(None)
            self.status_bar.showMessage("画布已清空")

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
    def _execute_flow(self):
        if not self.engine.nodes:
            QMessageBox.information(self, "提示", "画布中没有节点，请先添加算子。")
            return

        # 检查是否有输入节点
        has_input = any(
            len(n.inputs) == 0 for n in self.engine.nodes
        )
        if not has_input:
            QMessageBox.warning(self, "警告", "流程中缺少数据源节点（如「图像读取」）。")
            return

        self.status_bar.showMessage("正在执行流程...")
        QApplication.processEvents()

        try:
            results = self.engine.execute()
            self.status_bar.showMessage("流程执行完成 ✓")
            self._update_node_statuses()
            self.property_panel.refresh_output_preview()
        except Exception as e:
            self.status_bar.showMessage(f"执行失败: {e}")
            QMessageBox.critical(
                self, "执行错误",
                f"流程执行过程中发生错误:\n\n{traceback.format_exc()}"
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
            "<p><b>可用算子：</b></p>"
            "<ul>"
            "<li><b>图像读取</b> — 从文件加载图像，作为数据源</li>"
            "<li><b>灰度化</b> — 将彩色图像转换为灰度图</li>"
            "<li><b>高斯模糊</b> — 对图像进行高斯平滑</li>"
            "<li><b>边缘检测</b> — Canny 边缘提取</li>"
            "<li><b>阈值二值化</b> — 多种阈值处理方法</li>"
            "<li><b>图像缩放</b> — 调整图像尺寸</li>"
            "<li><b>结果输出</b> — 显示结果并可保存到文件</li>"
            "</ul>"
        )
        QMessageBox.information(self, "使用说明", text)


# ── 入口 ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ImageFlow")

    # 设置字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
