"""画布交互系统：基于 QGraphicsView/Scene 的节点编辑器。"""

from __future__ import annotations
import math
from typing import Any, Optional
from uuid import uuid4

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPainterPath,
    QLinearGradient, QRadialGradient,
)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsObject, QGraphicsPathItem,
    QGraphicsTextItem, QGraphicsProxyWidget, QMenu, QAction,
    QInputDialog, QMessageBox, QPushButton,
)

from node_base import Node, Port, Connection, PortType, ExecutionEngine

# ── 颜色定义 ──────────────────────────────────────────
COLOR_BG           = QColor(35, 35, 40)
COLOR_GRID         = QColor(50, 50, 55)
COLOR_NODE_BG      = QColor(55, 55, 65)
COLOR_NODE_BORDER  = QColor(80, 80, 95)
COLOR_NODE_TITLE   = QColor(65, 65, 80)
COLOR_PORT_IN      = QColor(220, 80, 80)
COLOR_PORT_OUT     = QColor(80, 160, 220)
COLOR_WIRE         = QColor(180, 180, 190)
COLOR_WIRE_ACTIVE  = QColor(255, 200, 50)
COLOR_TEXT         = QColor(220, 220, 225)
COLOR_TEXT_DIM     = QColor(150, 150, 160)
COLOR_TEXT_INPUT   = QColor(255, 100, 100)   # 输入节点 — 红色
COLOR_TEXT_OUTPUT  = QColor(100, 220, 130)   # 输出节点 — 绿色
COLOR_SELECTION    = QColor(100, 160, 255)
COLOR_SHADOW       = QColor(0, 0, 0, 60)

PORT_RADIUS = 6.0
NODE_MIN_WIDTH = 140.0
NODE_PORT_SPACING = 22.0
NODE_PADDING = 10.0
NODE_CORNER_RADIUS = 8.0
NODE_HEADER_H = 42.0


class NodeItem(QGraphicsObject):
    """画布上的节点图形项。"""

    # 信号由 canvas 监听
    node_moved = pyqtSignal(str)       # uid
    node_selected = pyqtSignal(str)

    def __init__(self, node: Node):
        super().__init__()
        w = max(NODE_MIN_WIDTH, self._calc_width(node))
        h = self._calc_height(node)
        self._rect = QRectF(0, 0, w, h)

        self.node = node
        self.port_items: dict[str, "PortItem"] = {}

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        self._build_header()
        self._build_ports()

    def boundingRect(self) -> QRectF:
        return self._rect

    def _calc_width(self, node: Node) -> float:
        return NODE_MIN_WIDTH

    def _calc_height(self, node: Node) -> float:
        n_ports = max(len(node.inputs), len(node.outputs))
        return NODE_HEADER_H + n_ports * NODE_PORT_SPACING + NODE_PADDING * 2

    def _build_header(self):
        """绘制节点头部标题。"""
        # 根据类别确定标题颜色
        cat_name = self.node.category
        if cat_name == "输入":
            title_color = COLOR_TEXT_INPUT
        elif cat_name == "输出":
            title_color = COLOR_TEXT_OUTPUT
        else:
            title_color = COLOR_TEXT

        title = QGraphicsTextItem(self.node.display_name, self)
        title.setDefaultTextColor(title_color)
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setPos(NODE_PADDING, 4)
        self._title_item = title

        cat = QGraphicsTextItem(cat_name, self)
        cat.setDefaultTextColor(COLOR_TEXT_DIM)
        cat.setFont(QFont("Microsoft YaHei", 8))
        cat.setPos(NODE_PADDING, 22)

    def _build_ports(self):
        """为每个端口创建 PortItem。键为 port.uid 以避免同名端口冲突。"""
        y_start = NODE_HEADER_H + 4.0
        r = self._rect

        for i, (__, port) in enumerate(self.node.inputs.items()):
            y = y_start + i * NODE_PORT_SPACING
            item = PortItem(port, self)
            item.setPos(0, y)
            self.port_items[port.uid] = item

        for i, (__, port) in enumerate(self.node.outputs.items()):
            y = y_start + i * NODE_PORT_SPACING
            item = PortItem(port, self)
            item.setPos(r.width(), y)
            self.port_items[port.uid] = item

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self._rect
        header_h = 32.0

        # 阴影
        shadow_rect = QRectF(rect.x() + 2, rect.y() + 2, rect.width(), rect.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(COLOR_SHADOW)
        painter.drawRoundedRect(shadow_rect, NODE_CORNER_RADIUS, NODE_CORNER_RADIUS)

        # 主体
        painter.setBrush(COLOR_NODE_BG)
        if self.isSelected():
            painter.setPen(QPen(COLOR_SELECTION, 2.5))
        else:
            painter.setPen(QPen(COLOR_NODE_BORDER, 1.5))
        painter.drawRoundedRect(rect, NODE_CORNER_RADIUS, NODE_CORNER_RADIUS)

        # 标题栏
        header_h = NODE_HEADER_H
        header_path = QPainterPath()
        header_path.moveTo(NODE_CORNER_RADIUS, 0)
        header_path.lineTo(rect.width() - NODE_CORNER_RADIUS, 0)
        header_path.arcTo(rect.width() - NODE_CORNER_RADIUS * 2, 0,
                          NODE_CORNER_RADIUS * 2, NODE_CORNER_RADIUS * 2, 90, -90)
        header_path.lineTo(rect.width(), header_h)
        header_path.lineTo(0, header_h)
        header_path.lineTo(0, NODE_CORNER_RADIUS)
        header_path.arcTo(0, 0, NODE_CORNER_RADIUS * 2, NODE_CORNER_RADIUS * 2, 180, -90)
        header_path.closeSubpath()

        painter.setPen(Qt.NoPen)
        painter.setBrush(COLOR_NODE_TITLE)
        painter.drawPath(header_path)

        # 分隔线
        painter.setPen(QPen(COLOR_NODE_BORDER, 0.5))
        painter.drawLine(QPointF(0, header_h), QPointF(rect.width(), header_h))

    def get_port_scene_pos(self, port: Port) -> QPointF:
        """获取指定 Port 对象在场景中的坐标。"""
        item = self.port_items.get(port.uid)
        if item:
            return self.mapToScene(item.center())
        return self.scenePos()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node_moved.emit(self.node.uid)
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.node_selected.emit(self.node.uid)
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = menu.addAction("删除节点")
        action = menu.exec_(event.screenPos())
        if action == remove_action:
            scene = self.scene()
            if isinstance(scene, NodeScene):
                scene.remove_node_item(self)


class PortItem(QGraphicsObject):
    """端口圆形指示器。"""

    port_clicked = pyqtSignal(Port)

    def __init__(self, port: Port, parent_node: NodeItem):
        super().__init__(parent_node)
        self.port = port
        self.node_item = parent_node
        self._radius = PORT_RADIUS
        self._rect = QRectF(-self._radius, -self._radius, self._radius * 2, self._radius * 2)

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)

        self._color = COLOR_PORT_OUT if port.port_type == PortType.OUTPUT else COLOR_PORT_IN
        self._border_color = self._color.darker(130)

        # 端口标签
        r = self._radius
        label = QGraphicsTextItem(port.name, self)
        label.setDefaultTextColor(COLOR_TEXT_DIM)
        label.setFont(QFont("Microsoft YaHei", 7))
        if port.port_type == PortType.OUTPUT:
            label.setPos(-label.boundingRect().width() - r - 3, -7)
        else:
            label.setPos(r + 4, -7)
        self._label = label

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(QPen(self._border_color, 1.5))
        painter.drawEllipse(self._rect)

    def center(self) -> QPointF:
        return self.pos()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.port_clicked.emit(self.port)
            event.accept()

    def mouseMoveEvent(self, event):
        scene = self.scene()
        if isinstance(scene, NodeScene) and scene._dragging_wire:
            scene._dragging_wire.update_end(event.scenePos())

    def mouseReleaseEvent(self, event):
        scene = self.scene()
        if isinstance(scene, NodeScene) and scene._dragging_wire and scene._drag_src_port:
            target_port = scene._find_port_at(event.scenePos())
            if target_port:
                scene._try_connect(scene._drag_src_port, target_port)
            scene._end_drag()

    def hoverEnterEvent(self, event):
        self.setScale(1.3)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)


class WireItem(QGraphicsPathItem):
    """两个端口之间的贝塞尔连线。"""

    def __init__(self, connection: Connection):
        super().__init__()
        self.connection = connection
        self.setPen(QPen(COLOR_WIRE, 2.0, Qt.SolidLine, Qt.RoundCap))
        self.setZValue(5)
        self.setAcceptHoverEvents(True)

    def update_path(self):
        src = self.connection.output_port
        tgt = self.connection.input_port
        src_node = self._find_node_item(src.node)
        tgt_node = self._find_node_item(tgt.node)

        if src_node and tgt_node:
            p1 = src_node.get_port_scene_pos(src)
            p2 = tgt_node.get_port_scene_pos(tgt)
        else:
            return

        dx = abs(p2.x() - p1.x()) * 0.5
        dx = max(dx, 50)

        path = QPainterPath()
        path.moveTo(p1)
        path.cubicTo(p1 + QPointF(dx, 0), p2 - QPointF(dx, 0), p2)
        self.setPath(path)

    def _find_node_item(self, node: Node) -> Optional[NodeItem]:
        scene = self.scene()
        if scene:
            for item in scene.items():
                if isinstance(item, NodeItem) and item.node is node:
                    return item
        return None

    def hoverEnterEvent(self, event):
        self.setPen(QPen(COLOR_WIRE_ACTIVE, 2.8, Qt.SolidLine, Qt.RoundCap))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(COLOR_WIRE, 2.0, Qt.SolidLine, Qt.RoundCap))
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = menu.addAction("删除连线")
        action = menu.exec_(event.screenPos())
        if action == remove_action:
            scene = self.scene()
            if isinstance(scene, NodeScene):
                scene.remove_wire_item(self)


class TempWireItem(QGraphicsPathItem):
    """拖拽中的临时连线。"""

    def __init__(self, start_pos: QPointF, is_from_output: bool):
        super().__init__()
        self._start = start_pos
        self._end = start_pos
        self._from_output = is_from_output
        self.setPen(QPen(QColor(255, 200, 50, 180), 2.0, Qt.DashLine, Qt.RoundCap))
        self.setZValue(100)

    def update_end(self, pos: QPointF):
        self._end = pos
        dx = abs(self._end.x() - self._start.x()) * 0.5
        dx = max(dx, 50)
        path = QPainterPath()
        if self._from_output:
            path.moveTo(self._start)
            path.cubicTo(self._start + QPointF(dx, 0),
                         self._end - QPointF(dx, 0), self._end)
        else:
            path.moveTo(self._end)
            path.cubicTo(self._end + QPointF(dx, 0),
                         self._start - QPointF(dx, 0), self._start)
        self.setPath(path)


class NodeScene(QGraphicsScene):
    """节点画布场景，处理节点/连线的交互逻辑。"""

    GRID_SIZE = 30
    SCENE_RECT = QRectF(-5000, -5000, 10000, 10000)

    def __init__(self, engine: ExecutionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.node_items: dict[str, NodeItem] = {}
        self.wire_items: dict[str, WireItem] = {}
        self._dragging_wire: Optional[TempWireItem] = None
        self._drag_src_port: Optional[Port] = None
        self.setSceneRect(self.SCENE_RECT)

    # ── 绘制背景网格 ──────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(rect, QBrush(COLOR_BG))

        pen = QPen(COLOR_GRID, 0.5)
        painter.setPen(pen)

        left = int(rect.left()) - (int(rect.left()) % self.GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % self.GRID_SIZE)

        lines = []
        for x in range(int(left), int(rect.right()), self.GRID_SIZE):
            lines.append((QPointF(x, rect.top()), QPointF(x, rect.bottom())))
        for y in range(int(top), int(rect.bottom()), self.GRID_SIZE):
            lines.append((QPointF(rect.left(), y), QPointF(rect.right(), y)))

        for p1, p2 in lines:
            painter.drawLine(p1, p2)

    # ── 节点管理 ──────────────────────────────────────
    def add_node(self, node_cls: type[Node], pos: QPointF = QPointF(0, 0)) -> Node:
        node = node_cls()
        self.engine.add_node(node)

        item = NodeItem(node)
        item.setPos(pos)
        self.addItem(item)
        self.node_items[node.uid] = item

        item.node_moved.connect(self._on_node_moved)
        for port_item in item.port_items.values():
            port_item.port_clicked.connect(self._on_port_clicked)

        return node

    def remove_node_item(self, item: NodeItem):
        # 移除关联连线
        to_remove = []
        for uid, wire in self.wire_items.items():
            conn = wire.connection
            if conn.output_port.node is item.node or conn.input_port.node is item.node:
                to_remove.append(uid)
        for uid in to_remove:
            self._remove_wire(uid)

        self.engine.remove_node(item.node)
        self.node_items.pop(item.node.uid, None)
        self.removeItem(item)

    # ── 连线管理 ──────────────────────────────────────
    def _on_port_clicked(self, port: Port):
        """点击端口开始拖拽连线。"""
        if self._dragging_wire is not None:
            return  # 已经有拖拽在进行

        item = self._find_port_item(port)
        if not item:
            return
        scene_pos = item.node_item.get_port_scene_pos(port)

        self._drag_src_port = port
        is_output = port.port_type == PortType.OUTPUT
        self._dragging_wire = TempWireItem(scene_pos, is_output)
        self.addItem(self._dragging_wire)

    def _find_port_item(self, port: Port) -> Optional[PortItem]:
        for ni in self.node_items.values():
            for pi in ni.port_items.values():
                if pi.port is port:
                    return pi
        return None

    def _find_port_at(self, scene_pos: QPointF) -> Optional[Port]:
        """在场景位置查找端口（排除拖拽起点的端口）。"""
        for ni in self.node_items.values():
            for pi in ni.port_items.values():
                if pi.port is self._drag_src_port:
                    continue
                port_scene_pos = ni.get_port_scene_pos(pi.port)
                dx = scene_pos.x() - port_scene_pos.x()
                dy = scene_pos.y() - port_scene_pos.y()
                if math.sqrt(dx * dx + dy * dy) < PORT_RADIUS * 2.5:
                    return pi.port
        return None

    def mouseMoveEvent(self, event):
        if self._dragging_wire:
            self._dragging_wire.update_end(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_wire and self._drag_src_port:
            target_port = self._find_port_at(event.scenePos())
            if target_port:
                self._try_connect(self._drag_src_port, target_port)
            self._end_drag()
        super().mouseReleaseEvent(event)

    def _try_connect(self, p1: Port, p2: Port):
        """尝试在两个端口间建立连接。"""
        # 确保方向正确：输出 → 输入
        if p1.port_type == p2.port_type:
            return
        out_port = p1 if p1.port_type == PortType.OUTPUT else p2
        in_port = p2 if p2.port_type == PortType.INPUT else p1

        # 检查是否已存在连接
        existing = self.engine.find_connection(out_port, in_port)
        if existing:
            return

        # 输入端口只能接收一个连接
        for c in self.engine.connections:
            if c.input_port is in_port:
                return

        conn = Connection(out_port, in_port)
        self.engine.add_connection(conn)

        wire = WireItem(conn)
        self.addItem(wire)
        wire.update_path()
        self.wire_items[conn.uid] = wire

    def _end_drag(self):
        if self._dragging_wire:
            self.removeItem(self._dragging_wire)
            self._dragging_wire = None
            self._drag_src_port = None

    def remove_wire_item(self, wire: WireItem):
        self._remove_wire(wire.connection.uid)

    def _remove_wire(self, uid: str):
        wire = self.wire_items.pop(uid, None)
        if wire:
            self.engine.remove_connection(wire.connection)
            self.removeItem(wire)

    def _on_node_moved(self, uid: str):
        """节点移动时更新所有关联连线的路径。"""
        node_item = self.node_items.get(uid)
        if not node_item:
            return
        for wire in self.wire_items.values():
            conn = wire.connection
            if conn.output_port.node.uid == uid or conn.input_port.node.uid == uid:
                wire.update_path()

    def update_all_wires(self):
        for wire in self.wire_items.values():
            wire.update_path()

    # ── 获取节点数据 ──────────────────────────────────
    def get_node(self, uid: str) -> Optional[Node]:
        item = self.node_items.get(uid)
        return item.node if item else None


class NodeCanvas(QGraphicsView):
    """画布视图，支持平移和缩放。"""

    min_zoom = 0.2
    max_zoom = 3.0
    zoom_factor = 1.15

    def __init__(self, scene: NodeScene, parent=None):
        super().__init__(scene, parent)
        self._scene = scene
        self.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("border: none; background: transparent;")

        self._pan = False
        self._pan_start = QPointF()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton
            and self.itemAt(event.pos()) is None
        ):
            self._pan = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan:
            self._pan = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = self.zoom_factor if delta > 0 else 1 / self.zoom_factor
        new_zoom = self.transform().m11() * factor

        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(factor, factor)
        event.accept()
