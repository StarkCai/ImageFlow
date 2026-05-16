"""区域读取算子：读取图像/视频，在预览框绘制区域（矩形/圆形/多边形），输出区域坐标和类型。"""

import os
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QPolygonF,
)
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem,
    QGraphicsLineItem,
    QListWidget, QListWidgetItem, QGroupBox, QSlider, QMessageBox,
    QWidget,
)

from node_base import Node
from node_registry import register_node
from nodes.image_input import _imread_unicode

SHAPE_RECT = "矩形"
SHAPE_CIRCLE = "圆形"
SHAPE_POLYGON = "多边形"

COLOR_RECT = QColor(255, 80, 80, 180)
COLOR_CIRCLE = QColor(80, 160, 255, 180)
COLOR_POLYGON = QColor(80, 255, 130, 180)
FILL_RECT = QColor(255, 80, 80, 40)
FILL_CIRCLE = QColor(80, 160, 255, 40)
FILL_POLYGON = QColor(80, 255, 130, 40)
COLOR_HANDLE = QColor(255, 255, 255)


def _ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """将 numpy 图像数组转换为 QPixmap（与 image_output 保持一致的实现）。"""
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── 控制手柄 ──────────────────────────────────────────
HANDLE_SIZE = 7.0


class HandleItem(QGraphicsRectItem):
    """可拖拽的控制手柄，用于编辑区域形状的角点/顶点。
    初始位置通过 rect 嵌入，避免 setPos 在 ItemSendsGeometryChanges 开启后触发递归。"""

    def __init__(self, px: float, py: float, parent_item: QGraphicsItem,
                 handle_idx: int, handle_color: QColor, dialog: "RegionDrawerDialog"):
        super().__init__(px - HANDLE_SIZE / 2, py - HANDLE_SIZE / 2,
                         HANDLE_SIZE, HANDLE_SIZE, parent_item)
        self._handle_idx = handle_idx
        self._dialog = dialog
        self.setPen(QPen(handle_color.darker(120), 1.5))
        self.setBrush(QColor(255, 255, 255, 220))
        self.setZValue(60)
        self.setCursor(Qt.SizeAllCursor)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            if not self._dialog._handles_updating:
                self._dialog._on_handle_moved(self)
        return super().itemChange(change, value)


# ── 绘图视图 ──────────────────────────────────────────
class DrawingView(QGraphicsView):
    """支持拖拽绘制的图像视图，鼠标左键绘制，中键平移，滚轮缩放。"""

    def __init__(self, scene: QGraphicsScene, dialog: "RegionDrawerDialog"):
        super().__init__(scene)
        self._dlg = dialog
        self._start_pos: Optional[QPointF] = None  # 场景坐标
        self._preview_item: Optional[QGraphicsItem] = None
        self._dragging = False
        self._panning = False
        self._pan_start = QPointF()

        self.setRenderHints(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("background: #1a1a20; border: 1px solid #444;")
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            # 检查是否点击在手柄上 — 如果是则交给手柄处理
            scene_pos = self.mapToScene(event.pos())
            item_at = self._dlg.scene().itemAt(scene_pos, self.transform())
            if isinstance(item_at, HandleItem):
                super().mousePressEvent(event)
                return

            self._start_pos = scene_pos
            shape = self._dlg.current_shape()

            if shape == SHAPE_POLYGON:
                self._dlg.add_polygon_vertex(scene_pos)
                self._dragging = False
            else:
                self._dragging = True
                self._create_preview(shape)
            event.accept()
            return
        if event.button() == Qt.RightButton:
            if self._dlg.current_shape() == SHAPE_POLYGON:
                self._dlg.finish_polygon()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        if self._dragging and self._preview_item and self._start_pos:
            scene_pos = self.mapToScene(event.pos())
            self._update_preview(scene_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            scene_pos = self.mapToScene(event.pos())
            self._finish_preview(scene_pos)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = self.transform().m11() * factor
        if 0.05 <= new_zoom <= 8.0:
            self.scale(factor, factor)
        event.accept()

    def _create_preview(self, shape: str):
        if self._preview_item:
            self.scene().removeItem(self._preview_item)

        if shape == SHAPE_RECT:
            item = QGraphicsRectItem()
            item.setPen(QPen(COLOR_RECT, 2))
            item.setBrush(FILL_RECT)
        elif shape == SHAPE_CIRCLE:
            item = QGraphicsEllipseItem()
            item.setPen(QPen(COLOR_CIRCLE, 2))
            item.setBrush(FILL_CIRCLE)
        else:
            return

        self.scene().addItem(item)
        self._preview_item = item

    def _update_preview(self, scene_pos: QPointF):
        if not self._preview_item or not self._start_pos:
            return
        shape = self._dlg.current_shape()
        x1, y1 = self._start_pos.x(), self._start_pos.y()
        x2, y2 = scene_pos.x(), scene_pos.y()

        if shape == SHAPE_RECT:
            rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self._preview_item.setRect(rect)
        elif shape == SHAPE_CIRCLE:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2
            r = max(rx, ry)
            self._preview_item.setRect(cx - r, cy - r, r * 2, r * 2)

    def _finish_preview(self, scene_pos: QPointF):
        if not self._preview_item or not self._start_pos:
            return
        shape = self._dlg.current_shape()
        if shape == SHAPE_RECT:
            rect = self._preview_item.rect()
            x1, y1 = int(rect.left()), int(rect.top())
            x2, y2 = int(rect.right()), int(rect.bottom())
            if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
                self.scene().removeItem(self._preview_item)
                self._preview_item = None
                return
            region = {"type": SHAPE_RECT,
                      "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            item = self._preview_item
        elif shape == SHAPE_CIRCLE:
            rect = self._preview_item.rect()
            cx = int(rect.center().x())
            cy = int(rect.center().y())
            r = int(rect.width() / 2)
            if r < 3:
                self.scene().removeItem(self._preview_item)
                self._preview_item = None
                return
            region = {"type": SHAPE_CIRCLE,
                      "coordinates": {"cx": cx, "cy": cy, "radius": r}}
            item = self._preview_item
        else:
            return

        self._preview_item = None
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.ItemIsMovable, True)
        self._dlg.add_region(region, item)

    def clear_drawing_state(self):
        if self._preview_item:
            self.scene().removeItem(self._preview_item)
            self._preview_item = None
        self._dragging = False
        self._start_pos = None


# ── 区域绘制对话框 ────────────────────────────────────
class RegionDrawerDialog(QDialog):
    """区域绘制对话框，在图像上绘制矩形/圆形/多边形。
    支持选择、拖拽移动、拖拽控制点编辑已有区域。"""

    def __init__(self, img: np.ndarray, title: str = "区域绘制",
                 existing_regions: List[dict] = None, parent=None):
        super().__init__(parent)
        self._img = img
        self._pixmap = _ndarray_to_qpixmap(img)
        self._current_shape = SHAPE_RECT
        self._regions: List[dict] = []
        self._region_items: list[QGraphicsItem] = []
        self._handle_items: list[HandleItem] = []

        # 多边形临时数据
        self._polygon_points: list[QPointF] = []
        self._polygon_vertex_items: list[QGraphicsEllipseItem] = []
        self._polygon_line_items: list[QGraphicsLineItem] = []

        # 当前选中的区域索引（用于高亮和编辑）
        self._selected_idx: int = -1

        # 防递归标志：手柄批量更新期间暂不响应单个手柄移动
        self._handles_updating: bool = False

        self.setWindowTitle(title)
        self.resize(1100, 720)
        self.setMinimumSize(800, 500)
        self._build_ui()

        if existing_regions:
            self._restore_regions(existing_regions)

    def scene(self) -> QGraphicsScene:
        return self._scene

    def current_shape(self) -> str:
        return self._current_shape

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 左侧：图形视图 ──────────────────────────
        self._scene = QGraphicsScene()
        self._pixmap_item = QGraphicsPixmapItem(self._pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(self._pixmap.rect()))
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

        self._view = DrawingView(self._scene, self)
        main_layout.addWidget(self._view, stretch=3)

        # ── 右侧：控制面板 ──────────────────────────
        right = QWidget()
        right.setFixedWidth(220)
        right.setStyleSheet("background: #2b2b30;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        shape_lbl = QLabel("形状类型")
        shape_lbl.setStyleSheet("color: #9a9aa0; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(shape_lbl)

        self._shape_combo = QComboBox()
        self._shape_combo.addItems([SHAPE_RECT, SHAPE_CIRCLE, SHAPE_POLYGON])
        self._shape_combo.setStyleSheet(
            "QComboBox { background: #3d3d45; color: #dcdce0; border: 1px solid #555; "
            "border-radius: 4px; padding: 6px 10px; font-size: 12px; }"
            "QComboBox:hover { border-color: #777; }"
            "QComboBox QAbstractItemView { background: #3d3d45; color: #dcdce0; "
            "selection-background-color: #5a9cf8; border: 1px solid #555; }"
        )
        self._shape_combo.currentTextChanged.connect(self._on_shape_changed)
        right_layout.addWidget(self._shape_combo)

        self._hint = QLabel("按住左键拖拽绘制矩形\n中键拖拽平移，滚轮缩放\n选中区域后可拖拽移动/编辑控制点")
        self._hint.setStyleSheet("color: #707078; font-size: 10px; padding: 4px 0;")
        self._hint.setWordWrap(True)
        right_layout.addWidget(self._hint)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #444;")
        right_layout.addWidget(sep)

        list_title = QLabel("已绘制区域")
        list_title.setStyleSheet("color: #9a9aa0; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(list_title)

        self._region_list = QListWidget()
        self._region_list.setStyleSheet(
            "QListWidget { background: #1e1e24; color: #dcdce0; border: 1px solid #444; "
            "border-radius: 4px; font-size: 11px; padding: 2px; }"
            "QListWidget::item { padding: 3px 6px; }"
            "QListWidget::item:selected { background: #5a9cf8; }"
        )
        self._region_list.itemClicked.connect(self._on_list_item_clicked)
        right_layout.addWidget(self._region_list)

        btn_row = QHBoxLayout()
        del_btn = QPushButton("删除选中")
        del_btn.setStyleSheet(
            "QPushButton { background: #e05560; color: #fff; border: none; "
            "border-radius: 3px; padding: 5px 10px; font-size: 10px; }"
            "QPushButton:hover { background: #f0707a; }"
        )
        del_btn.clicked.connect(self._delete_region)
        btn_row.addWidget(del_btn)

        clear_btn = QPushButton("清空全部")
        clear_btn.setStyleSheet(
            "QPushButton { background: #555; color: #dcdce0; border: none; "
            "border-radius: 3px; padding: 5px 10px; font-size: 10px; }"
            "QPushButton:hover { background: #666; }"
        )
        clear_btn.clicked.connect(self._clear_regions)
        btn_row.addWidget(clear_btn)
        right_layout.addLayout(btn_row)

        right_layout.addStretch()

        action_row = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #3d3d45; color: #dcdce0; border: 1px solid #555; "
            "border-radius: 4px; padding: 8px 16px; font-size: 12px; }"
            "QPushButton:hover { background: #4d4d58; }"
        )
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(cancel_btn)

        ok_btn = QPushButton("确定")
        ok_btn.setStyleSheet(
            "QPushButton { background: #5a9cf8; color: #fff; border: none; "
            "border-radius: 4px; padding: 8px 16px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background: #7ab4ff; }"
        )
        ok_btn.clicked.connect(self.accept)
        action_row.addWidget(ok_btn)
        right_layout.addLayout(action_row)

        main_layout.addWidget(right)

    # ── 形状切换 ──────────────────────────────────
    def _on_shape_changed(self, text: str):
        self._view.clear_drawing_state()
        self._current_shape = text
        self._cancel_polygon()
        hints = {
            SHAPE_RECT: "按住左键拖拽绘制矩形\n中键拖拽平移，滚轮缩放\n选中区域后可拖拽移动/编辑控制点",
            SHAPE_CIRCLE: "按住左键拖拽绘制圆形\n中键拖拽平移，滚轮缩放\n选中区域后可拖拽移动/编辑控制点",
            SHAPE_POLYGON: "左键添加顶点，右键闭合完成\n中键拖拽平移，滚轮缩放",
        }
        self._hint.setText(hints.get(text, ""))

    # ── 多边形 ────────────────────────────────────
    def add_polygon_vertex(self, pos: QPointF):
        r = 4
        dot = QGraphicsEllipseItem(pos.x() - r, pos.y() - r, r * 2, r * 2)
        dot.setPen(QPen(QColor(255, 200, 50), 2))
        dot.setBrush(QColor(255, 200, 50))
        self._scene.addItem(dot)
        self._polygon_vertex_items.append(dot)

        if len(self._polygon_points) >= 1:
            prev = self._polygon_points[-1]
            line = QGraphicsLineItem(QLineF(prev, pos))
            line.setPen(QPen(QColor(255, 200, 50, 160), 1.5, Qt.DashLine))
            self._scene.addItem(line)
            self._polygon_line_items.append(line)

        self._polygon_points.append(pos)

    def finish_polygon(self):
        if len(self._polygon_points) < 3:
            return

        polygon = QPolygonF([QPointF(p) for p in self._polygon_points])
        item = QGraphicsPolygonItem(polygon)
        item.setPen(QPen(COLOR_POLYGON, 2))
        item.setBrush(FILL_POLYGON)
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.ItemIsMovable, True)
        self._scene.addItem(item)

        points = [(int(p.x()), int(p.y())) for p in self._polygon_points]
        region = {"type": SHAPE_POLYGON, "coordinates": {"points": points}}
        self.add_region(region, item)
        self._cancel_polygon()

    def _cancel_polygon(self):
        for dot in self._polygon_vertex_items:
            self._scene.removeItem(dot)
        self._polygon_vertex_items.clear()
        for line in self._polygon_line_items:
            self._scene.removeItem(line)
        self._polygon_line_items.clear()
        self._polygon_points.clear()

    # ── 场景选中 → 控制手柄 ──────────────────────
    def _on_scene_selection_changed(self):
        self._remove_all_handles()
        selected = self._scene.selectedItems()
        if not selected:
            self._selected_idx = -1
            self._region_list.clearSelection()
            return

        item = selected[0]
        # 查找此 item 在 region_items 中的索引
        try:
            idx = self._region_items.index(item)
        except ValueError:
            self._selected_idx = -1
            return

        self._selected_idx = idx
        # 同步列表选中
        self._region_list.blockSignals(True)
        self._region_list.setCurrentRow(idx)
        self._region_list.blockSignals(False)
        # 同步因 ItemIsMovable 产生的位置偏移
        self._sync_region_from_item(idx)
        self._create_handles(item, idx)
        self._update_region_list()

    def _on_list_item_clicked(self, list_item: QListWidgetItem):
        row = self._region_list.row(list_item)
        if 0 <= row < len(self._region_items):
            item = self._region_items[row]
            self._scene.clearSelection()
            item.setSelected(True)
            self._selected_idx = row
            self._remove_all_handles()
            self._sync_region_from_item(row)
            self._create_handles(item, row)
            self._update_region_list()

    # ── 手柄管理 ──────────────────────────────────
    def _remove_all_handles(self):
        for h in self._handle_items:
            self._scene.removeItem(h)
        self._handle_items.clear()

    def _create_handles(self, item: QGraphicsItem, idx: int):
        self._remove_all_handles()
        region = self._regions[idx]
        rtype = region["type"]
        coords = region["coordinates"]

        if rtype == SHAPE_RECT:
            pts = [(coords["x1"], coords["y1"]),
                   (coords["x2"], coords["y1"]),
                   (coords["x1"], coords["y2"]),
                   (coords["x2"], coords["y2"])]
        elif rtype == SHAPE_CIRCLE:
            cx, cy, r = coords["cx"], coords["cy"], coords["radius"]
            pts = [(cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)]
        elif rtype == SHAPE_POLYGON:
            pts = [(p[0], p[1]) for p in coords["points"]]
        else:
            return

        shape_color = self._shape_color(rtype)
        for i, (px, py) in enumerate(pts):
            h = HandleItem(px, py, item, i, shape_color, self)
            self._handle_items.append(h)

    def _shape_color(self, rtype: str) -> QColor:
        if rtype == SHAPE_RECT:
            return COLOR_RECT
        elif rtype == SHAPE_CIRCLE:
            return COLOR_CIRCLE
        return COLOR_POLYGON

    def _on_handle_moved(self, handle: HandleItem):
        """手柄拖动后，更新区域图形和数据。"""
        if self._handles_updating:
            return
        self._handles_updating = True
        try:
            self._do_handle_moved(handle)
        finally:
            self._handles_updating = False

    def _do_handle_moved(self, handle: HandleItem):
        idx = self._selected_idx
        if idx < 0 or idx >= len(self._regions):
            return
        item = self._region_items[idx]
        region = self._regions[idx]
        rtype = region["type"]

        # 获取手柄在父坐标下的实际位置（rect 中心 + pos 偏移）
        rc = handle.rect().center()
        hp = handle.pos() + rc
        hx, hy = hp.x(), hp.y()
        hi = handle._handle_idx

        if rtype == SHAPE_RECT:
            coords = region["coordinates"]
            if hi == 0:      # top-left
                coords["x1"], coords["y1"] = int(hx), int(hy)
            elif hi == 1:    # top-right
                coords["x2"], coords["y1"] = int(hx), int(hy)
            elif hi == 2:    # bottom-left
                coords["x1"], coords["y2"] = int(hx), int(hy)
            elif hi == 3:    # bottom-right
                coords["x2"], coords["y2"] = int(hx), int(hy)
            x1, y1 = coords["x1"], coords["y1"]
            x2, y2 = coords["x2"], coords["y2"]
            item.setRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

        elif rtype == SHAPE_CIRCLE:
            coords = region["coordinates"]
            cx, cy = coords["cx"], coords["cy"]
            if hi == 0:      # left
                r = abs(cx - int(hx))
            elif hi == 1:    # right
                r = abs(int(hx) - cx)
            elif hi == 2:    # top
                r = abs(cy - int(hy))
            else:            # bottom
                r = abs(int(hy) - cy)
            r = max(r, 3)
            coords["radius"] = r
            item.setRect(cx - r, cy - r, r * 2, r * 2)

        elif rtype == SHAPE_POLYGON:
            coords = region["coordinates"]
            pts = coords["points"]
            if 0 <= hi < len(pts):
                pts[hi] = (int(hx), int(hy))
            poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
            item.setPolygon(poly)

        # 先移除旧手柄，避免 _sync_region_from_item 中 setPos() 触发
        # 手柄的 itemChange → _on_handle_moved 导致递归
        self._remove_all_handles()
        self._sync_region_from_item(idx)
        self._create_handles(item, idx)
        self._update_region_list()

    def _sync_region_from_item(self, idx: int):
        """将 QGraphicsItem 的当前状态同步回 self._regions[idx]。"""
        if idx < 0 or idx >= len(self._regions):
            return
        item = self._region_items[idx]
        region = self._regions[idx]
        rtype = region["type"]

        if rtype == SHAPE_RECT:
            # 考虑 item 的位置偏移（Moveable）
            rect = item.rect()
            offset = item.pos()
            coords = region["coordinates"]
            coords["x1"] = int(rect.left() + offset.x())
            coords["y1"] = int(rect.top() + offset.y())
            coords["x2"] = int(rect.right() + offset.x())
            coords["y2"] = int(rect.bottom() + offset.y())
            # 重置 pos 并更新 rect
            item.setPos(0, 0)
            item.setRect(QRectF(coords["x1"], coords["y1"],
                                coords["x2"] - coords["x1"],
                                coords["y2"] - coords["y1"]))

        elif rtype == SHAPE_CIRCLE:
            rect = item.rect()
            offset = item.pos()
            coords = region["coordinates"]
            coords["cx"] = int(rect.center().x() + offset.x())
            coords["cy"] = int(rect.center().y() + offset.y())
            coords["radius"] = int(rect.width() / 2)
            item.setPos(0, 0)
            cx, cy, r = coords["cx"], coords["cy"], coords["radius"]
            item.setRect(cx - r, cy - r, r * 2, r * 2)

        elif rtype == SHAPE_POLYGON:
            offset = item.pos()
            poly = item.polygon()
            coords = region["coordinates"]
            pts = []
            for i in range(poly.count()):
                p = poly.at(i)
                pts.append((int(p.x() + offset.x()),
                            int(p.y() + offset.y())))
            coords["points"] = pts
            item.setPos(0, 0)
            new_poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
            item.setPolygon(new_poly)

    # ── 区域管理 ──────────────────────────────────
    def _restore_regions(self, regions: List[dict]):
        """恢复之前绘制的区域到场景和列表中。"""
        for region in regions:
            rtype = region["type"]
            coords = region["coordinates"]
            item = None

            if rtype == SHAPE_RECT:
                x1, y1 = coords["x1"], coords["y1"]
                x2, y2 = coords["x2"], coords["y2"]
                item = QGraphicsRectItem(min(x1, x2), min(y1, y2),
                                         abs(x2 - x1), abs(y2 - y1))
                item.setPen(QPen(COLOR_RECT, 2))
                item.setBrush(FILL_RECT)
            elif rtype == SHAPE_CIRCLE:
                cx, cy, r = coords["cx"], coords["cy"], coords["radius"]
                item = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2)
                item.setPen(QPen(COLOR_CIRCLE, 2))
                item.setBrush(FILL_CIRCLE)
            elif rtype == SHAPE_POLYGON:
                pts = coords["points"]
                polygon = QPolygonF([QPointF(p[0], p[1]) for p in pts])
                item = QGraphicsPolygonItem(polygon)
                item.setPen(QPen(COLOR_POLYGON, 2))
                item.setBrush(FILL_POLYGON)

            if item:
                item.setFlag(QGraphicsItem.ItemIsSelectable, True)
                item.setFlag(QGraphicsItem.ItemIsMovable, True)
                self._scene.addItem(item)
                self._regions.append(region)
                self._region_items.append(item)
        self._update_region_list()

    def add_region(self, region: dict, item: QGraphicsItem):
        self._regions.append(region)
        self._region_items.append(item)
        self._update_region_list()

    def _update_region_list(self):
        self._region_list.clear()
        for i, r in enumerate(self._regions):
            rtype = r["type"]
            if rtype == SHAPE_RECT:
                c = r["coordinates"]
                detail = f"({c['x1']},{c['y1']})-({c['x2']},{c['y2']})"
            elif rtype == SHAPE_CIRCLE:
                c = r["coordinates"]
                detail = f"({c['cx']},{c['cy']}) r={c['radius']}"
            elif rtype == SHAPE_POLYGON:
                detail = f"{len(r['coordinates']['points'])} 顶点"
            else:
                detail = ""
            self._region_list.addItem(f"{i + 1}. {rtype}  {detail}")

    def _delete_region(self):
        row = self._region_list.currentRow()
        if 0 <= row < len(self._regions):
            self._remove_all_handles()
            self._regions.pop(row)
            item = self._region_items.pop(row)
            self._scene.removeItem(item)
            self._selected_idx = -1
            self._update_region_list()

    def _clear_regions(self):
        self._remove_all_handles()
        self._regions.clear()
        for item in self._region_items:
            self._scene.removeItem(item)
        self._region_items.clear()
        self._selected_idx = -1
        self._update_region_list()

    def get_regions(self) -> List[dict]:
        return self._regions

    # ── 右键 ──────────────────────────────────────
    def contextMenuEvent(self, event):
        if self._current_shape == SHAPE_POLYGON and len(self._polygon_points) >= 3:
            self.finish_polygon()
            event.accept()
            return
        super().contextMenuEvent(event)


# ── 区域读取算子 ──────────────────────────────────────
@register_node
class RegionInputNode(Node):
    display_name = "区域读取"
    category = "输入"

    def __init__(self):
        self._file_path: str = ""
        self._regions: List[dict] = []
        self._preview_image: Optional[np.ndarray] = None
        super().__init__()

    def _setup_ports(self):
        self.add_output("区域")

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, path: str):
        self._file_path = path

    @property
    def region_count(self) -> int:
        return len(self._regions)

    def process(self, **inputs):
        if not self._file_path:
            raise ValueError("未设置图像/视频文件路径")
        if not self._regions:
            raise ValueError("未绘制区域，请在属性面板中点击「绘制区域」")
        return {"区域": self._regions}

    def open_draw_dialog(self):
        if not self._file_path:
            QMessageBox.information(None, "提示", "请先选择图像或视频文件。")
            return

        img = self._load_image()
        if img is None:
            QMessageBox.warning(None, "错误", f"无法读取文件:\n{self._file_path}")
            return

        self._preview_image = img
        title = f"区域绘制 — {os.path.basename(self._file_path)}"
        dlg = RegionDrawerDialog(img, title=title,
                                 existing_regions=list(self._regions))
        if dlg.exec_() == QDialog.Accepted:
            self._regions = dlg.get_regions()

    def _load_image(self) -> Optional[np.ndarray]:
        ext = os.path.splitext(self._file_path)[1].lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        if ext in video_exts:
            cap = cv2.VideoCapture(self._file_path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        else:
            img = _imread_unicode(self._file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
