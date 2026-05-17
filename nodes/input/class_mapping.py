"""类别映射算子：读取本地 JSON 文件，将类别 ID 映射为名称。"""

import json
import os
from typing import Optional

from node_base import Node
from node_registry import register_node


CLASS_MAPPING_REFERENCE = """
标准类别映射格式 (JSON):
{
    "0": "head",
    "1": "person",
    "2": "car"
}
键为类别 ID (数字字符串)，值为类别名称。
""".strip()


def _validate_mapping(data) -> Optional[dict]:
    """验证并返回标准化的类别映射，格式不符返回 None。"""
    if not isinstance(data, dict):
        return None
    mapping = {}
    for k, v in data.items():
        if not isinstance(v, str):
            return None
        try:
            int(k)
        except (ValueError, TypeError):
            return None
        mapping[str(k)] = v
    return mapping


@register_node
class ClassMappingNode(Node):
    display_name = "类别映射"
    category = "输入"
    algorithms = ["类别映射"]

    def __init__(self):
        self.json_path: str = ""
        self._mapping: dict = {}
        self._error_msg: str = ""
        super().__init__()

    def _setup_ports(self):
        self.add_output("类别映射", data_type="类别映射")

    def _load_mapping(self):
        """加载并验证 JSON 文件。"""
        if not self.json_path:
            raise ValueError("未指定类别映射文件")
        if not os.path.exists(self.json_path):
            raise ValueError(f"类别映射文件不存在: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping = _validate_mapping(data)
        if mapping is None:
            raise ValueError(
                f"类别映射文件格式不正确。\n期望格式:\n{CLASS_MAPPING_REFERENCE}"
            )
        self._mapping = mapping
        self._error_msg = ""

    def process(self, **inputs):
        try:
            self._load_mapping()
        except (ValueError, json.JSONDecodeError, OSError) as e:
            self._error_msg = str(e)
            self._mapping = {}
            raise

        # 构建摘要供日志输出
        preview = ", ".join(
            f"{k}:{v}" for k, v in list(self._mapping.items())[:10]
        )
        if len(self._mapping) > 10:
            preview += f" ... 共 {len(self._mapping)} 类"
        self._result_summary = f"加载类别映射: {preview}"

        return {"类别映射": {"mapping": self._mapping}}
