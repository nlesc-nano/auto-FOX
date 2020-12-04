"""A module containing the :class:`UniqueLoader` class."""

from collections.abc import Hashable
from typing import Dict, Any

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore[misc]
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

__all__ = ['UniqueLoader']


class UniqueLoader(Loader):
    """A special :class:`~yaml.Loader` with duplicate key checking."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> Dict[Any, Any]:
        """Construct Convert the passed **node** into a :class:`dict`."""
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None, f"expected a mapping node, but found {node.id}", node.start_mark
            )

        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark,
                                       "found unhashable key", key_node.start_mark)
            elif key in mapping:
                raise ConstructorError("while constructing a mapping", node.start_mark,
                                       "found a duplicate key", key_node.start_mark)

            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping
