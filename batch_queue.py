"""批处理队列：BatchItem 数据容器和 ImageQueue 有界队列。"""

from __future__ import annotations
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BatchItem:
    """单个批处理图像的数据容器。"""
    image: np.ndarray       # RGB uint8 HxWx3
    filename: str           # 基础文件名，如 "photo.jpg"
    filepath: str           # 完整路径，用于错误报告
    index: int              # 0-based 序号


class ImageQueue:
    """有界图像队列，生产者-消费者模型，内置进度和错误追踪。

    None 值表示流结束哨兵。
    """

    def __init__(self, maxsize: int = 5):
        self._queue: queue.Queue[Optional[BatchItem]] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self.total: int = 0
        self.consumed: int = 0
        self._cancelled: bool = False
        self._errors: list[tuple[int, str, str]] = []  # (index, filepath, message)

    def put(self, item: Optional[BatchItem]):
        """入队一个图像（阻塞直到有空间）。None 结束流。"""
        self._queue.put(item)

    def put_nowait(self, item: Optional[BatchItem]):
        """非阻塞入队；队列满时触发 queue.Full。"""
        self._queue.put_nowait(item)

    def get(self, timeout: Optional[float] = None) -> Optional[BatchItem]:
        """出队一个图像（阻塞直到有数据）。返回 None 表示流结束。"""
        item = self._queue.get(timeout=timeout)
        if item is not None:
            with self._lock:
                self.consumed += 1
        return item

    def cancel(self):
        """取消队列，推送哨兵以解除阻塞等待的消费者。"""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def add_error(self, index: int, filepath: str, message: str):
        """线程安全地记录文件级错误。"""
        with self._lock:
            self._errors.append((index, filepath, message))

    @property
    def is_done(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def progress(self) -> float:
        with self._lock:
            if self.total == 0:
                return 0.0
            return min(self.consumed / self.total, 1.0)

    @property
    def errors(self) -> list[tuple[int, str, str]]:
        with self._lock:
            return list(self._errors)
