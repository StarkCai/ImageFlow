"""节点注册管理器，维护所有可用的节点类型。"""

from __future__ import annotations
from typing import Type
from node_base import Node


class NodeRegistry:
    """单例模式的节点注册表。"""

    _instance: NodeRegistry | None = None

    def __new__(cls) -> NodeRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: dict[str, Type[Node]] = {}
        return cls._instance

    def register(self, node_cls: Type[Node]):
        self._registry[node_cls.display_name] = node_cls

    def get(self, name: str) -> Type[Node] | None:
        return self._registry.get(name)

    def list_all(self) -> dict[str, Type[Node]]:
        return dict(self._registry)

    def list_by_category(self) -> dict[str, list[Type[Node]]]:
        cats: dict[str, list[Type[Node]]] = {}
        for cls in self._registry.values():
            cats.setdefault(cls.category, []).append(cls)
        return cats


def register_node(cls: Type[Node]) -> Type[Node]:
    """装饰器：将节点类注册到全局注册表。"""
    NodeRegistry().register(cls)
    return cls
