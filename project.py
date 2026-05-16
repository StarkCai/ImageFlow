"""工程文件管理：保存 / 加载节点流程为 JSON 文件。"""

from __future__ import annotations
import json
from typing import Any

from node_base import Node

# 框架属性，不参与序列化
_SKIP_ATTRS = {
    "uid", "inputs", "outputs", "_data", "_processed",
    "display_name", "category", "algorithms",
    "_result_summary", "_last_regions", "_last_json",
    "_last_image", "_mapping", "_error_msg",
}

PROJECT_VERSION = "1.0"


def _node_params(node: Node) -> dict:
    """提取节点可序列化的参数。"""
    params = {}
    for k, v in node.__dict__.items():
        if k.startswith("_") or k in _SKIP_ATTRS:
            continue
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict)):
            if isinstance(v, tuple):
                params[k] = list(v)
            else:
                params[k] = v
    return params


def apply_params(node: Node, params: dict):
    """将保存的参数恢复到节点实例。"""
    for k, v in params.items():
        if hasattr(node, k):
            current = getattr(node, k)
            if isinstance(current, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(node, k, v)


def save_project(engine, scene, filepath: str):
    """将当前引擎和场景状态保存为 JSON 工程文件。"""
    nodes_data = []
    for node in engine.nodes:
        item = scene.node_items.get(node.uid)
        pos = item.pos() if item else None
        nodes_data.append({
            "uid": node.uid,
            "type": node.__class__.__name__,
            "x": pos.x() if pos else 0.0,
            "y": pos.y() if pos else 0.0,
            "params": _node_params(node),
        })

    connections_data = []
    for conn in engine.connections:
        connections_data.append({
            "uid": conn.uid,
            "src_node": conn.output_port.node.uid,
            "src_port": conn.output_port.name,
            "tgt_node": conn.input_port.node.uid,
            "tgt_port": conn.input_port.name,
        })

    project = {
        "version": PROJECT_VERSION,
        "nodes": nodes_data,
        "connections": connections_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)


def load_project(filepath: str) -> dict:
    """从 JSON 文件加载工程数据，返回原始字典。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
