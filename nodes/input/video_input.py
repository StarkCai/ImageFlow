"""视频读取算子：支持单文件/文件夹视频帧读取，生产者-消费者队列批处理。"""

from __future__ import annotations
import os
import threading
from typing import Optional

import cv2
import numpy as np

from node_base import Node
from node_registry import register_node
from batch_queue import BatchItem, ImageQueue

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")


@register_node
class VideoInputNode(Node):
    display_name = "视频读取"
    category = "输入"

    def __init__(self):
        # ── 可序列化配置 ──
        self.file_path: str = ""
        self.input_mode: str = "single"     # "single" | "folder"
        self.folder_path: str = ""
        self.recursive: bool = False
        self.frame_skip: int = 0
        self.queue_size: int = 5

        # ── 运行时状态（不序列化）──
        self._batch_queue: Optional[ImageQueue] = None
        self._batch_files: list[str] = []
        self._current_batch_filename: str = ""
        self._current_batch_index: int = -1
        self._current_video_name: str = ""

        super().__init__()

    def _setup_ports(self):
        self.add_output("图像")

    # ── 视频文件发现 ──────────────────────────────

    def _discover_files(self) -> list[str]:
        """在文件夹中发现视频文件，返回排序后的绝对路径列表。"""
        files: list[str] = []
        folder = self.folder_path

        if self.recursive:
            for root, _, filenames in os.walk(folder):
                for fn in filenames:
                    if fn.lower().endswith(_VIDEO_EXTENSIONS):
                        files.append(os.path.join(root, fn))
        else:
            try:
                for fn in os.listdir(folder):
                    if fn.lower().endswith(_VIDEO_EXTENSIONS):
                        files.append(os.path.join(folder, fn))
            except OSError:
                pass

        files.sort()
        return files

    # ── 批处理生命周期 ──────────────────────────────

    def prepare_batch(self) -> int:
        """准备批处理：发现视频文件，创建队列，启动生产者线程。返回总帧数。"""
        if self.input_mode == "single":
            if not self.file_path:
                raise ValueError("未设置视频文件路径")
            if not os.path.isfile(self.file_path):
                raise ValueError(f"视频文件不存在: {self.file_path}")
            self._batch_files = [self.file_path]
        else:
            if not self.folder_path:
                raise ValueError("未设置视频文件夹路径")
            self._batch_files = self._discover_files()
            if not self._batch_files:
                raise ValueError(f"文件夹中未发现视频文件: {self.folder_path}")

        # 估算总帧数用于进度显示
        total_frames = 0
        for f in self._batch_files:
            cap = cv2.VideoCapture(f)
            if cap.isOpened():
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames += max(1, (fc + self.frame_skip) // (self.frame_skip + 1))
            cap.release()

        self._batch_queue = ImageQueue(maxsize=self.queue_size)
        self._batch_queue.total = total_frames

        t = threading.Thread(target=self._producer_loop, daemon=True)
        t.start()
        return total_frames

    def _producer_loop(self):
        """生产者线程：逐一打开视频文件，按帧跳过策略读取帧并放入队列。"""
        global_idx = 0
        for vid_idx, filepath in enumerate(self._batch_files):
            if self._batch_queue.is_done:
                break

            video_name = os.path.splitext(os.path.basename(filepath))[0]
            self._current_video_name = video_name

            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                self._batch_queue.add_error(
                    vid_idx, filepath, f"无法打开视频: {filepath}"
                )
                continue

            try:
                frame_idx = 0   # 原始帧序号
                kept_count = 0  # 保留下来的帧序号

                while not self._batch_queue.is_done:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 应用帧跳过：每 (frame_skip+1) 帧保留一帧
                    if frame_idx % (self.frame_skip + 1) == 0:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        filename = f"{video_name}_frame_{kept_count:06d}.jpg"
                        item = BatchItem(
                            image=img_rgb,
                            filename=filename,
                            filepath=filepath,
                            index=global_idx,
                        )
                        self._batch_queue.put(item)
                        kept_count += 1
                        global_idx += 1

                    frame_idx += 1

            except Exception as e:
                self._batch_queue.add_error(vid_idx, filepath, str(e))
            finally:
                cap.release()

        self._batch_queue.put(None)  # 流结束哨兵

    def next_batch_item(self, timeout: Optional[float] = None) -> Optional[BatchItem]:
        """获取下一帧图像。返回 None 表示批处理结束。"""
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
        self._current_video_name = ""

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

    # ── 核心处理（单文件模式）───────────────────────

    def process(self, **inputs):
        if self.input_mode == "folder":
            raise RuntimeError(
                "文件夹模式需通过引擎的 execute_batch() 方法执行"
            )

        if not self.file_path:
            raise ValueError("未设置视频文件路径")

        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {self.file_path}")

        try:
            # 跳过前 frame_skip 帧，读取下一帧
            skip_count = self.frame_skip
            while skip_count > 0:
                ret, _ = cap.read()
                if not ret:
                    break
                skip_count -= 1

            ret, frame = cap.read()
            if not ret:
                # 回退：尝试读取第一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"视频无可用帧: {self.file_path}")
        finally:
            cap.release()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_video_name = os.path.splitext(
            os.path.basename(self.file_path))[0]
        return {"图像": img_rgb}
