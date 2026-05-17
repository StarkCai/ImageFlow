"""节点基础框架：端口、节点、连线的抽象基类。"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from uuid import uuid4


class PortType(Enum):
    INPUT = auto()
    OUTPUT = auto()


@dataclass
class Port:
    """节点的输入或输出端口。"""
    name: str
    port_type: PortType
    node: "Node"
    data_type: str = ""
    multi_connect: bool = False
    uid: str = field(default_factory=lambda: uuid4().hex[:8])

    def __post_init__(self):
        if not self.data_type:
            self.data_type = self.name

    def __hash__(self):
        return hash(self.uid)

    def __repr__(self):
        return f"Port({self.name}, {self.port_type.name})"


@dataclass
class Connection:
    """两个端口之间的连线。"""
    output_port: Port
    input_port: Port
    uid: str = field(default_factory=lambda: uuid4().hex[:8])

    def __post_init__(self):
        if self.output_port.port_type != PortType.OUTPUT:
            raise ValueError("connection source must be an output port")
        if self.input_port.port_type != PortType.INPUT:
            raise ValueError("connection target must be an input port")


class Node:
    """所有算子的基类。"""

    display_name: str = "Node"
    category: str = "General"

    def __init__(self):
        self.uid: str = uuid4().hex[:8]
        self.inputs: dict[str, Port] = {}
        self.outputs: dict[str, Port] = {}
        self._setup_ports()
        self._data: dict[str, Any] = {}
        self._processed: bool = False

    def _setup_ports(self):
        """子类重写以定义输入/输出端口。"""

    def add_input(self, name: str, data_type: str = "",
                  multi_connect: bool = False) -> Port:
        port = Port(name, PortType.INPUT, self,
                    data_type=data_type, multi_connect=multi_connect)
        self.inputs[name] = port
        return port

    def add_output(self, name: str, data_type: str = "") -> Port:
        port = Port(name, PortType.OUTPUT, self, data_type=data_type)
        self.outputs[name] = port
        return port

    def process(self, **inputs: Any) -> dict[str, Any]:
        """执行算子逻辑，子类必须实现。"""
        raise NotImplementedError

    def set_data(self, port_name: str, value: Any):
        self._data[port_name] = value

    def get_data(self, port_name: str) -> Any:
        return self._data.get(port_name)

    def reset(self):
        self._data.clear()
        self._processed = False


class ExecutionEngine:
    """节点图执行引擎，按拓扑顺序执行节点。

    支持两种模式：
    - execute()：单次执行，所有节点各运行一次
    - execute_batch()：批处理执行，对文件夹中的每张图像重复运行管道
    """

    def __init__(self):
        self.nodes: list[Node] = []
        self.connections: list[Connection] = []
        self._progress_callback: Optional[Callable[[str, str], None]] = None
        self._cancelled: bool = False

    def add_node(self, node: Node):
        self.nodes.append(node)

    def remove_node(self, node: Node):
        self.nodes.remove(node)
        self.connections = [
            c for c in self.connections
            if c.output_port.node is not node and c.input_port.node is not node
        ]

    def add_connection(self, conn: Connection):
        if conn not in self.connections:
            self.connections.append(conn)

    def remove_connection(self, conn: Connection):
        if conn in self.connections:
            self.connections.remove(conn)

    def find_connection(self, out_port: Port, in_port: Port) -> Optional[Connection]:
        for c in self.connections:
            if c.output_port is out_port and c.input_port is in_port:
                return c
        return None

    def _topological_sort(self) -> list[Node]:
        """返回拓扑排序后的节点列表，源节点在前。"""
        in_degree: dict[str, int] = {n.uid: 0 for n in self.nodes}
        adj: dict[str, list[Node]] = {n.uid: [] for n in self.nodes}

        for conn in self.connections:
            src_uid = conn.output_port.node.uid
            tgt_uid = conn.input_port.node.uid
            in_degree[tgt_uid] += 1
            adj[src_uid].append(conn.input_port.node)

        queue = [n for n in self.nodes if in_degree[n.uid] == 0]
        result: list[Node] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj[node.uid]:
                in_degree[neighbor.uid] -= 1
                if in_degree[neighbor.uid] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise RuntimeError("graph contains a cycle, cannot execute")

        return result

    # ── 输入收集 ──────────────────────────────────────

    def _gather_inputs(self, node: Node) -> dict[str, Any]:
        """收集节点所有输入端口的连接数据。"""
        input_data: dict[str, Any] = {}

        for name, in_port in node.inputs.items():
            if in_port.multi_connect:
                source_list = []
                for conn in self.connections:
                    if conn.input_port is in_port:
                        src_node = conn.output_port.node
                        src_port_name = conn.output_port.name
                        source_list.append(src_node.get_data(src_port_name))
                input_data[name] = source_list
            else:
                source_data = None
                for conn in self.connections:
                    if conn.input_port is in_port:
                        src_node = conn.output_port.node
                        src_port_name = conn.output_port.name
                        source_data = src_node.get_data(src_port_name)
                        break
                input_data[name] = source_data

        return input_data

    # ── 单次执行 ──────────────────────────────────────

    def execute(self) -> dict[str, Any]:
        """执行整个节点图，返回每个节点的输出结果。"""
        for node in self.nodes:
            node.reset()

        order = self._topological_sort()
        results: dict[str, Any] = {}

        for node in order:
            input_data = self._gather_inputs(node)

            cb = self._progress_callback
            if cb:
                cb(node.display_name, "start")

            try:
                outputs = node.process(**input_data)
                if outputs:
                    for port_name, value in outputs.items():
                        node.set_data(port_name, value)
            except Exception as e:
                if cb:
                    cb(node.display_name, f"error: {e}")
                raise

            if cb:
                cb(node.display_name, "done")

            results[node.uid] = {
                "node": node.display_name,
                "outputs": {k: type(v).__name__ for k, v in node._data.items()},
            }

        return results

    # ── 批处理执行 ────────────────────────────────────

    def _find_batch_source_nodes(self) -> list:
        """查找所有需要批处理执行的源节点（文件夹模式或视频文件）。"""
        result = []
        for node in self.nodes:
            if len(node.inputs) == 0:
                if node.__class__.__name__ in ("ImageInputNode", "VideoInputNode"):
                    if getattr(node, "input_mode", "single") == "folder":
                        result.append(node)
                    elif node.__class__.__name__ == "VideoInputNode":
                        # 单视频文件模式也走批处理，以处理所有帧
                        if getattr(node, "file_path", ""):
                            result.append(node)
        return result

    def _post_batch_iteration(self, item):
        """批处理迭代成功后，处理输出节点的自动保存。"""
        import os as _os
        for node in self.nodes:
            if node.__class__.__name__ == "ImageOutputNode":
                save_dir = getattr(node, "save_dir", "")
                if save_dir and getattr(node, "_last_image", None) is not None:
                    _os.makedirs(save_dir, exist_ok=True)
                    save_path = _os.path.join(save_dir, item.filename)
                    if not _os.path.splitext(save_path)[1]:
                        save_path += ".png"
                    node.save_last_result(save_path)

            if node.__class__.__name__ == "RegionOutputNode":
                save_json = getattr(node, "save_json", False)
                save_dir = getattr(node, "save_dir", "")
                if save_json and save_dir and getattr(node, "_last_json", ""):
                    _os.makedirs(save_dir, exist_ok=True)
                    base = _os.path.splitext(item.filename)[0]
                    save_path = _os.path.join(save_dir, base + ".json")
                    node.save_last_json(save_path)

            if node.__class__.__name__ == "VideoOutputNode":
                output_mode = getattr(node, "output_mode", "image")
                auto_save = getattr(node, "auto_save", False)
                img = getattr(node, "_last_image", None)
                if img is None:
                    continue
                if output_mode == "video":
                    # 从源节点获取当前视频名
                    for sn in self.nodes:
                        if len(sn.inputs) == 0 and \
                           sn.__class__.__name__ == "VideoInputNode":
                            new_name = getattr(sn, "_current_video_name", "")
                            node._current_source_name = new_name
                            break
                    # 检测视频切换：新视频开始时先完成上一个视频的写入
                    prev = getattr(node, "_previous_source_name", "")
                    cur = getattr(node, "_current_source_name", "")
                    if prev and cur and prev != cur:
                        try:
                            node.finalize_batch()
                        except Exception:
                            pass
                    node._previous_source_name = cur
                    if auto_save:
                        node.append_frame(img)
                elif output_mode == "image":
                    if auto_save:
                        save_dir = getattr(node, "save_dir", "")
                        if save_dir:
                            _os.makedirs(save_dir, exist_ok=True)
                            base = _os.path.splitext(item.filename)[0]
                            ext = getattr(node, "image_format", "jpg")
                            save_path = _os.path.join(save_dir, base + "." + ext)
                            node.save_last_result(save_path)

    def execute_batch(self) -> dict[str, Any]:
        """批处理入口：检测到文件夹源节点时，循环执行管道。

        如果没有文件夹模式的源节点，回退到单次 execute()。
        """
        source_nodes = self._find_batch_source_nodes()
        if not source_nodes:
            return self.execute()

        self._cancelled = False
        source_node = source_nodes[0]

        try:
            total = source_node.prepare_batch()
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        cb = self._progress_callback
        if cb:
            cb("批次", f"start_batch_{total}")

        order = self._topological_sort()
        order_without_source = [n for n in order if n is not source_node]

        completed = 0
        while not self._cancelled:
            item = source_node.next_batch_item()
            if item is None:
                break

            # 每轮迭代重置除源节点外的所有节点
            for node in order_without_source:
                node.reset()

            # 将源图像注入源节点
            source_node.set_data("图像", item.image)
            source_node._processed = True

            if cb:
                cb("批次", f"start_image_{item.index + 1}_{total}_{item.filename}")

            iteration_aborted = False
            for node in order_without_source:
                if self._cancelled:
                    iteration_aborted = True
                    break

                input_data = self._gather_inputs(node)

                if cb:
                    cb(node.display_name, "start")

                try:
                    outputs = node.process(**input_data)
                    if outputs:
                        for port_name, value in outputs.items():
                            node.set_data(port_name, value)
                except Exception as e:
                    if cb:
                        cb(node.display_name, f"error: {e}")
                    iteration_aborted = True
                    break

                if cb:
                    cb(node.display_name, "done")

            if not iteration_aborted:
                try:
                    self._post_batch_iteration(item)
                except Exception:
                    pass  # 保存失败不中断批处理

            completed += 1

            if cb:
                cb("批次", f"done_image_{completed}_{total}")

        batch_errors = source_node.batch_errors
        source_node.cleanup_batch()

        # 批处理后处理：视频输出节点最终化
        for node in self.nodes:
            if node.__class__.__name__ == "VideoOutputNode":
                finalize = getattr(node, "finalize_batch", None)
                if finalize:
                    try:
                        finalize()
                    except Exception as e:
                        if cb:
                            cb(node.display_name, f"error: 视频写入失败 - {e}")

        if cb:
            cb("批次", "end_batch")

        return {
            "mode": "batch",
            "total": total,
            "completed": completed,
            "errors": batch_errors,
        }

    def cancel(self):
        """取消正在进行的批处理。"""
        self._cancelled = True
        for node in self.nodes:
            if hasattr(node, "cancel_batch"):
                node.cancel_batch()


# ── 区域标准格式工具 ──────────────────────────────────
#
# 标准输出格式 (region_input / object_detection 等节点遵循):
#   {"regions": [
#       {"id": 1, "type": "矩形", "class": None, "ocr": "", "coordinates": {"x1":..., "y1":..., "x2":..., "y2":...}},
#       {"id": 2, "type": "圆形", "class": None, "ocr": "", "coordinates": {"cx":..., "cy":..., "radius":...}},
#       {"id": 3, "type": "多边形", "class": None, "ocr": "Hello", "coordinates": {"points": [[x1,y1],...]}},
#   ], "width": 1920, "height": 1080}
#
# 消费者节点 (overlay / crop 等) 通过 _extract_regions() 兼容新旧格式。


def format_regions(regions: list, width: int = 0, height: int = 0) -> dict:
    """将区域列表包装为标准输出格式，自动分配 id (从 1 开始)。

    width/height 指明区域所处的坐标系尺寸。0 表示未指定。
    """
    out = []
    for i, r in enumerate(regions):
        item = {
            "id": i + 1,
            "type": r["type"],
            "class": r.get("class_id") if "class_id" in r else r.get("class"),
            "ocr": r.get("ocr", ""),
            "coordinates": r["coordinates"],
        }
        out.append(item)
    return {"regions": out, "width": width, "height": height}


def _extract_regions(data) -> list:
    """从各种可能的区域数据格式中提取区域列表（兼容新旧格式与连线方式）。

    支持:
      - 标准格式: {"regions": [{...}, ...]}
      - 旧格式(单区域): {"type": "...", "coordinates": {...}}
      - 旧格式(列表): [{"type": "...", ...}, ...]
      - 多连线列表: [{"regions": [...]}, ...] 或 [[...], [...]]
    """
    if not data:
        return []
    if isinstance(data, dict):
        if "regions" in data:
            return data["regions"]
        if "type" in data:
            return [data]
        return []
    if isinstance(data, list):
        flat = []
        for item in data:
            if isinstance(item, dict):
                if "regions" in item:
                    flat.extend(item["regions"])
                elif "type" in item:
                    flat.append(item)
            elif isinstance(item, list):
                flat.extend(item)
        return flat
    return []


def _extract_region_dims(data) -> tuple:
    """从区域数据中提取坐标系尺寸 (width, height)。

    返回 (0, 0) 表示未携带尺寸信息（旧格式或未知坐标系）。
    """
    if isinstance(data, dict):
        return data.get("width", 0), data.get("height", 0)
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            return data[0].get("width", 0), data[0].get("height", 0)
    return 0, 0
