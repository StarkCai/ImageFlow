"""点云输出算子：双面板点云查看器（左侧 3D 视图 + 右侧点信息）并支持保存。"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QFont,
)
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QSplitter,
    QLabel, QGroupBox, QFormLayout, QPushButton, QFileDialog,
    QMessageBox, QSizePolicy,
)

from node_base import Node
from node_registry import register_node


def _project_points(points: np.ndarray, eye, center, up,
                    fov: float, width: int, height: int) -> tuple:
    """将 3D 点投影到 2D 屏幕坐标，返回 (pixel_coords, depths)。"""
    # 相机坐标系
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    # 相对于相机的位置
    rel = points - eye
    # 点积到相机坐标系
    z = rel.dot(forward)  # depth along view direction
    x = rel.dot(right)
    y = rel.dot(cam_up)

    # 透视投影
    fov_rad = np.radians(fov) / 2.0
    scale = 1.0 / np.tan(fov_rad)
    aspect = width / height

    # 齐次裁剪坐标
    px = (x / z) * scale * aspect
    py = (y / z) * scale

    # 屏幕坐标（原点在中心）
    sx = (px + 1.0) * 0.5 * width
    sy = (1.0 - py) * 0.5 * height

    return np.column_stack([sx, sy]), z


def _pick_point_kdtree(points_3d: np.ndarray, sx: float, sy: float,
                       eye, center, up, fov: float,
                       width: int, height: int,
                       pick_radius_px: float = 8.0) -> int:
    """屏幕坐标反投影 → KDTree 近邻搜索，返回最近点索引。"""
    import open3d as o3d

    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    fov_rad = np.radians(fov) / 2.0
    scale = 1.0 / np.tan(fov_rad)
    aspect = width / height

    # 屏幕 → 归一化设备坐标
    ndc_x = (sx / width - 0.5) * 2.0
    ndc_y = (0.5 - sy / height) * 2.0

    # 反投影 → 相机空间方向
    cam_x = ndc_x / (scale * aspect)
    cam_y = ndc_y / scale
    ray_dir = forward + cam_x * right + cam_y * cam_up
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # KDTree 搜索
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points_3d)
    tree = o3d.geometry.KDTreeFlann(pcd_tmp)

    # 沿射线步进搜索
    max_dist = float(np.linalg.norm(points_3d.max(axis=0) - points_3d.min(axis=0)) * 3)
    if max_dist < 0.1:
        max_dist = 10.0
    pick_radius = pick_radius_px * max_dist / max(width, height)

    best_idx, best_dist = -1, float("inf")
    for t in np.linspace(0.01, max_dist, 200):
        pt = eye + ray_dir * t
        _, idx, dists = tree.search_knn_vector_3d(pt, 1)
        d = float(np.sqrt(dists[0]))
        if d < pick_radius and d < best_dist:
            best_dist = d
            best_idx = idx[0]

    return best_idx if best_dist < pick_radius else -1


class PointCloudCanvas(QLabel):
    """点云 3D 画布：使用 QPainter 软件渲染点云投影 + 鼠标交互。"""

    def __init__(self, pcd):
        super().__init__()
        import open3d as o3d
        self._pcd = pcd
        self._points = np.asarray(pcd.points).astype(np.float64)
        self._colors = (np.asarray(pcd.colors) if pcd.has_colors()
                        else np.full((len(self._points), 3), [0.7, 0.7, 0.7]))

        # 相机
        pts_min = self._points.min(axis=0)
        pts_max = self._points.max(axis=0)
        self._center = (pts_min + pts_max) / 2.0
        self._radius = float(np.linalg.norm(pts_max - pts_min) * 1.5)
        if self._radius < 0.01:
            self._radius = 1.0
        self._theta: float = 30.0
        self._phi: float = 25.0
        self._fov: float = 50.0

        # 鼠标状态
        self._last_pos: Optional[QPoint] = None
        self._picked_idx: int = -1
        self._point_size: int = 3

        # 显示子采样（加速渲染，不影响拾取精度）
        self._max_draw: int = 100000
        self._draw_indices: Optional[np.ndarray] = None

        self.setMinimumSize(500, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

        # 信号：通知外部选中点变化
        self.point_picked = None  # 由外部设置回调: fn(idx, x, y, z)

    # ── 相机 ───────────────────────────────────────────

    def _get_camera(self):
        t = np.radians(self._theta)
        p = np.radians(self._phi)
        eye = self._center + self._radius * np.array([
            np.cos(t) * np.cos(p),
            np.sin(t) * np.cos(p),
            np.sin(p),
        ])
        up = np.array([0.0, 0.0, 1.0])
        if abs(self._phi) > 85:
            up = np.array([0.0, -np.sign(self._phi), 0.0])
        return eye, self._center, up

    # ── 渲染 ───────────────────────────────────────────

    def _render_to_image(self, w: int, h: int) -> QImage:
        """将点云渲染到 QImage 缓冲区（直接像素写入，快于逐点 QPainter）。"""
        eye, center, up = self._get_camera()

        if len(self._points) > self._max_draw:
            if self._draw_indices is None or len(self._draw_indices) != self._max_draw:
                rng = np.random.RandomState(42)
                self._draw_indices = rng.choice(
                    len(self._points), self._max_draw, replace=False)
            pts = self._points[self._draw_indices]
            clr = self._colors[self._draw_indices]
        else:
            pts = self._points
            clr = self._colors

        proj, depths = _project_points(pts, eye, center, up, self._fov, w, h)

        # 深度排序，远点先写入
        order = np.argsort(-depths)

        # RGBA 缓冲区 (H, W, 4) uint8
        buf = np.zeros((h, w, 4), dtype=np.uint8)
        buf[:, :, 3] = 255

        # 写入点 — 直接操作 numpy 数组，无 Python 逐点循环
        ix = np.clip(np.round(proj[order, 0]).astype(np.int32), 0, w - 1)
        iy = np.clip(np.round(proj[order, 1]).astype(np.int32), 0, h - 1)
        rgb = np.clip(clr[order] * 255, 0, 255).astype(np.uint8)

        ps = self._point_size
        if ps <= 1:
            buf[iy, ix, :3] = rgb
        else:
            r = ps // 2
            for dy in range(-r, r + 1):
                yy = np.clip(iy + dy, 0, h - 1)
                for dx in range(-r, r + 1):
                    xx = np.clip(ix + dx, 0, w - 1)
                    buf[yy, xx, :3] = rgb

        # 绘制选中点十字标记
        if self._picked_idx >= 0:
            picked_pt = self._points[self._picked_idx:self._picked_idx + 1]
            pp, _ = _project_points(picked_pt, eye, center, up, self._fov, w, h)
            px, py = int(round(pp[0, 0])), int(round(pp[0, 1]))
            s = 12
            for dx in range(-s, s + 1):
                x = np.clip(px + dx, 0, w - 1)
                y = np.clip(py, 0, h - 1)
                buf[y, x] = [0, 255, 100, 255]
                x = np.clip(px, 0, w - 1)
                y = np.clip(py + dx, 0, h - 1)
                buf[y, x] = [0, 255, 100, 255]

        img = QImage(buf.data, w, h, w * 4, QImage.Format_RGBA8888)
        # 数据必须保持在 QImage 生命周期内
        img._buf_ref = buf
        return img

    def paintEvent(self, event):
        super().paintEvent(event)
        w, h = self.width(), self.height()

        if w < 10 or h < 10 or len(self._points) == 0:
            painter = QPainter(self)
            painter.fillRect(0, 0, w, h, QColor(26, 26, 32))
            painter.setPen(QColor(120, 120, 140))
            painter.drawText(self.rect(), Qt.AlignCenter, "无点云数据")
            painter.end()
            return

        qimg = self._render_to_image(w, h)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(self)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

    # ── 鼠标 ───────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last_pos is not None:
            dx = event.x() - self._last_pos.x()
            dy = event.y() - self._last_pos.y()
            self._theta -= dx * 0.3
            self._phi += dy * 0.3
            self._phi = max(-89.0, min(89.0, self._phi))
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._last_pos is not None:
            dx = abs(event.x() - self._last_pos.x())
            dy = abs(event.y() - self._last_pos.y())
            if dx < 3 and dy < 3:
                self._do_pick(event.x(), event.y())
        self._last_pos = None
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._radius *= (0.9 if delta > 0 else 1.1)
        self._radius = max(0.001, min(1000.0, self._radius))
        self.update()

    def _do_pick(self, sx: int, sy: int):
        w, h = self.width(), self.height()
        eye, center, up = self._get_camera()
        idx = _pick_point_kdtree(self._points, sx, sy, eye, center, up,
                                 self._fov, w, h)
        self._picked_idx = idx
        if self.point_picked is not None:
            if idx >= 0:
                pt = self._points[idx]
                self.point_picked(idx, float(pt[0]), float(pt[1]), float(pt[2]))
            else:
                self.point_picked(-1, 0, 0, 0)
        self.update()

    # ── 公共接口 ───────────────────────────────────────

    def set_view(self, theta: float, phi: float):
        self._theta = theta
        self._phi = phi
        self._picked_idx = -1
        self.update()

    def set_point_size(self, size: int):
        self._point_size = max(1, min(10, size))
        self.update()


class PointCloudViewerDialog(QDialog):
    """点云预览对话框：左侧 3D 视图 + 右侧信息/控制面板。"""

    def __init__(self, pcd, parent=None):
        super().__init__(parent)
        self.setWindowTitle("点云预览")
        self.resize(1280, 800)
        self.setMinimumSize(900, 550)
        self._build_ui(pcd)

    def _build_ui(self, pcd):
        central = QSplitter(Qt.Horizontal)

        # ── 左侧：点云画布 ──────────────────────────
        self._canvas = PointCloudCanvas(pcd)
        self._canvas.point_picked = self._on_point_picked
        central.addWidget(self._canvas)

        # ── 右侧：信息面板 ──────────────────────────
        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(350)
        rlayout = QVBoxLayout(right)
        rlayout.setContentsMargins(12, 12, 12, 12)
        rlayout.setSpacing(10)

        label_style = "color: #5a9cf8; font-size: 18px; font-weight: bold;"
        group_style = (
            "QGroupBox { font-size: 13px; font-weight: bold; color: #dcdce0; "
            "border: 1px solid #444; border-radius: 6px; margin-top: 12px; "
            "padding: 16px 12px 12px 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }"
        )

        # 选中点信息
        info_group = QGroupBox("选中点信息")
        info_group.setStyleSheet(group_style)
        form = QFormLayout()
        form.setSpacing(8)

        self._idx_label = QLabel("—")
        self._idx_label.setStyleSheet("color: #9a9aa0; font-size: 12px;")
        form.addRow("序号:", self._idx_label)

        self._x_label = QLabel("—")
        self._x_label.setStyleSheet(label_style)
        form.addRow("X:", self._x_label)

        self._y_label = QLabel("—")
        self._y_label.setStyleSheet(label_style)
        form.addRow("Y:", self._y_label)

        self._z_label = QLabel("—")
        self._z_label.setStyleSheet(label_style)
        form.addRow("Z:", self._z_label)

        pts = np.asarray(pcd.points)
        self._has_colors = pcd.has_colors()
        if self._has_colors:
            self._r_label = QLabel("—")
            self._g_label = QLabel("—")
            self._b_label = QLabel("—")
            form.addRow("R:", self._r_label)
            form.addRow("G:", self._g_label)
            form.addRow("B:", self._b_label)
            self._colors_np = np.asarray(pcd.colors)
        else:
            self._r_label = self._g_label = self._b_label = None
            self._colors_np = None

        info_group.setLayout(form)
        rlayout.addWidget(info_group)

        # 操作提示
        hint = QLabel(
            "操作:\n"
            "  拖拽鼠标 → 旋转视角\n"
            "  滚轮     → 缩放\n"
            "  点击     → 选取点"
        )
        hint.setStyleSheet("color: #6a6a75; font-size: 11px; line-height: 1.6;")
        hint.setWordWrap(True)
        rlayout.addWidget(hint)

        # 点大小
        ps_layout = QHBoxLayout()
        ps_layout.addWidget(QLabel("点大小:"))
        ps_layout.addStretch()
        for s in [1, 2, 3, 5, 7]:
            btn = QPushButton(str(s))
            btn.setFixedSize(30, 26)
            btn.setStyleSheet(
                "QPushButton { background: #3d3d45; color: #dcdce0; "
                "border: none; border-radius: 3px; font-size: 11px; }"
                "QPushButton:hover { background: #5a9cf8; }"
            )
            btn.clicked.connect(lambda _, sz=s: self._canvas.set_point_size(sz))
            ps_layout.addWidget(btn)
        rlayout.addLayout(ps_layout)

        # 视角按钮
        views = QGroupBox("快速视角")
        views.setStyleSheet(group_style)
        vlayout = QVBoxLayout()
        row1 = QHBoxLayout()
        for name, th, ph in [("前", 0, 0), ("后", 180, 0),
                              ("左", -90, 0), ("右", 90, 0)]:
            btn = QPushButton(name)
            btn.setFixedHeight(32)
            btn.setStyleSheet(
                "QPushButton { background: #3d3d45; color: #dcdce0; "
                "border: none; border-radius: 3px; font-size: 12px; }"
                "QPushButton:hover { background: #5a9cf8; }"
            )
            btn.clicked.connect(lambda _, t=th, p=ph: self._canvas.set_view(t, p))
            row1.addWidget(btn)
        vlayout.addLayout(row1)
        row2 = QHBoxLayout()
        for name, th, ph in [("上", 0, 89), ("下", 0, -89), ("复位", 30, 25)]:
            btn = QPushButton(name)
            btn.setFixedHeight(32)
            btn.setStyleSheet(
                "QPushButton { background: #3d3d45; color: #dcdce0; "
                "border: none; border-radius: 3px; font-size: 12px; }"
                "QPushButton:hover { background: #5a9cf8; }"
            )
            btn.clicked.connect(lambda _, t=th, p=ph: self._canvas.set_view(t, p))
            row2.addWidget(btn)
        vlayout.addLayout(row2)
        views.setLayout(vlayout)
        rlayout.addWidget(views)

        # 点云统计
        pts_arr = np.asarray(pcd.points)
        stats_group = QGroupBox("点云统计")
        stats_group.setStyleSheet(group_style)
        sform = QFormLayout()
        sform.setSpacing(6)
        sform.addRow("总点数:", QLabel(f"{len(pts_arr):,}"))
        bbox = (
            f"[{pts_arr[:,0].min():.3f}, {pts_arr[:,1].min():.3f}, {pts_arr[:,2].min():.3f}]\n"
            f"→ [{pts_arr[:,0].max():.3f}, {pts_arr[:,1].max():.3f}, {pts_arr[:,2].max():.3f}]"
        )
        bb_label = QLabel(bbox)
        bb_label.setWordWrap(True)
        bb_label.setStyleSheet("color: #9a9aa0; font-size: 10px;")
        sform.addRow("包围盒:", bb_label)
        stats_group.setLayout(sform)
        rlayout.addWidget(stats_group)

        rlayout.addStretch()
        central.addWidget(right)
        central.setStretchFactor(0, 3)
        central.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(central)

    def _on_point_picked(self, idx: int, x: float, y: float, z: float):
        if idx >= 0:
            self._idx_label.setText(str(idx))
            self._x_label.setText(f"{x:.6f}")
            self._y_label.setText(f"{y:.6f}")
            self._z_label.setText(f"{z:.6f}")
            if self._has_colors and self._colors_np is not None:
                c = self._colors_np[idx]
                self._r_label.setText(f"{c[0]:.4f}")
                self._g_label.setText(f"{c[1]:.4f}")
                self._b_label.setText(f"{c[2]:.4f}")
        else:
            self._idx_label.setText("未命中")
            self._x_label.setText("—")
            self._y_label.setText("—")
            self._z_label.setText("—")
            if self._has_colors:
                self._r_label.setText("—")
                self._g_label.setText("—")
                self._b_label.setText("—")


@register_node
class PointCloudOutputNode(Node):
    display_name = "点云输出"
    category = "输出"

    def __init__(self):
        self.save_dir: str = ""
        self._last_pointcloud = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("点云", data_type="点云")

    def process(self, **inputs):
        pcd = inputs.get("点云")
        if pcd is None:
            raise ValueError("未接收到点云数据")

        self._last_pointcloud = pcd
        return {"点云": pcd}

    def show_last_result(self):
        """弹出双面板点云预览对话框。"""
        pcd = self._last_pointcloud
        if pcd is None:
            raise ValueError("没有可显示的点云，请先执行流程")

        viewer = PointCloudViewerDialog(pcd)
        viewer.exec_()

    def save_last_result(self, filepath: str):
        """保存点云到文件。"""
        import open3d as o3d

        pcd = self._last_pointcloud
        if pcd is None:
            raise ValueError("没有可保存的点云，请先执行流程")

        success = o3d.io.write_point_cloud(filepath, pcd)
        if not success:
            raise RuntimeError(f"无法保存点云到: {filepath}")
