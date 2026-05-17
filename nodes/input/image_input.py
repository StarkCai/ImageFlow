"""图像读取算子：支持单文件读取与文件夹批量读取（含中文路径）。"""

from __future__ import annotations
import os
import threading
from typing import Optional

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node
from batch_queue import BatchItem, ImageQueue

_VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")


def _imread_unicode(filepath: str) -> Optional[np.ndarray]:
    """支持中文路径的图像读取。"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@register_node
class ImageInputNode(Node):
    display_name = "图像读取"
    category = "输入"

    def __init__(self):
        # ── 可序列化配置 ──
        self.file_path: str = ""
        self.input_mode: str = "single"     # "single" | "folder"
        self.folder_path: str = ""
        self.recursive: bool = False
        self.queue_size: int = 5

        # ── 运行时状态（不序列化）──
        self._batch_queue: Optional[ImageQueue] = None
        self._batch_files: list[str] = []
        self._current_batch_filename: str = ""
        self._current_batch_index: int = -1

        super().__init__()

    def _setup_ports(self):
        self.add_output("图像")

    # ── 文件夹文件发现 ──────────────────────────────

    def _discover_files(self) -> list[str]:
        """在文件夹中发现图像文件，返回排序后的绝对路径列表。"""
        files: list[str] = []
        folder = self.folder_path

        if self.recursive:
            for root, _, filenames in os.walk(folder):
                for fn in filenames:
                    if fn.lower().endswith(_VALID_EXTENSIONS):
                        files.append(os.path.join(root, fn))
        else:
            try:
                for fn in os.listdir(folder):
                    if fn.lower().endswith(_VALID_EXTENSIONS):
                        files.append(os.path.join(folder, fn))
            except OSError:
                pass

        files.sort()
        return files

    # ── 批处理生命周期 ──────────────────────────────

    def prepare_batch(self) -> int:
        """准备批处理：发现文件，创建队列，启动生产者线程。返回文件数。"""
        if not self.folder_path:
            raise ValueError("未设置图像文件夹路径")

        self._batch_files = self._discover_files()
        if not self._batch_files:
            raise ValueError(f"文件夹中未发现图像文件: {self.folder_path}")

        self._batch_queue = ImageQueue(maxsize=self.queue_size)
        self._batch_queue.total = len(self._batch_files)

        t = threading.Thread(target=self._producer_loop, daemon=True)
        t.start()
        return len(self._batch_files)

    def _producer_loop(self):
        """生产者线程：从磁盘读取图像并放入队列。"""
        for idx, filepath in enumerate(self._batch_files):
            if self._batch_queue.is_done:
                break

            try:
                img = _imread_unicode(filepath)
                if img is None:
                    self._batch_queue.add_error(
                        idx, filepath, f"无法读取图像: {filepath}"
                    )
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                item = BatchItem(
                    image=img_rgb,
                    filename=os.path.basename(filepath),
                    filepath=filepath,
                    index=idx,
                )
                self._batch_queue.put(item)

            except Exception as e:
                self._batch_queue.add_error(idx, filepath, str(e))

        self._batch_queue.put(None)  # 流结束哨兵

    def next_batch_item(self, timeout: Optional[float] = None) -> Optional[BatchItem]:
        """获取下一张图像。返回 None 表示批处理结束。"""
        if self._batch_queue is None:
            return None

        item = self._batch_queue.get(timeout=timeout)
        if item is not None:
            self._current_batch_filename = item.filename
            self._current_batch_index = item.index
        return item

    def cancel_batch(self):
        """取消批处理。"""
        if self._batch_queue is not None:
            self._batch_queue.cancel()

    def cleanup_batch(self):
        """清理批处理状态。"""
        self._batch_queue = None
        self._batch_files.clear()
        self._current_batch_filename = ""
        self._current_batch_index = -1

    @property
    def batch_progress(self) -> float:
        if self._batch_queue is None:
            return 0.0
        return self._batch_queue.progress

    @property
    def batch_errors(self) -> list[tuple[int, str, str]]:
        if self._batch_queue is None:
            return []
        return self._batch_queue.errors

    # ── 核心处理 ────────────────────────────────────

    def process(self, **inputs):
        if self.input_mode == "folder":
            raise RuntimeError(
                "文件夹模式需通过引擎的 execute_batch() 方法执行"
            )

        if not self.file_path:
            raise ValueError("未设置图像文件路径")

        img = _imread_unicode(self.file_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {self.file_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {"图像": img_rgb}
