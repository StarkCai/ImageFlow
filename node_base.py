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
    uid: str = field(default_factory=lambda: uuid4().hex[:8])

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

    def add_input(self, name: str) -> Port:
        port = Port(name, PortType.INPUT, self)
        self.inputs[name] = port
        return port

    def add_output(self, name: str) -> Port:
        port = Port(name, PortType.OUTPUT, self)
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
    """节点图执行引擎，按拓扑顺序执行节点。"""

    def __init__(self):
        self.nodes: list[Node] = []
        self.connections: list[Connection] = []

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

    def execute(self) -> dict[str, Any]:
        """执行整个节点图，返回每个节点的输出结果。"""
        for node in self.nodes:
            node.reset()

        order = self._topological_sort()
        results: dict[str, Any] = {}

        for node in order:
            input_data: dict[str, Any] = {}

            for name, in_port in node.inputs.items():
                # 查找连到此输入端口的连接
                source_data = None
                for conn in self.connections:
                    if conn.input_port is in_port:
                        src_node = conn.output_port.node
                        src_port_name = conn.output_port.name
                        source_data = src_node.get_data(src_port_name)
                        break
                input_data[name] = source_data

            try:
                outputs = node.process(**input_data)
                if outputs:
                    for port_name, value in outputs.items():
                        node.set_data(port_name, value)
            except Exception as e:
                results[f"__error__{node.uid}"] = str(e)

            results[node.uid] = {
                "node": node.display_name,
                "outputs": {k: type(v).__name__ for k, v in node._data.items()},
            }

        return results
