"""
FOX.classes.frozen_settings
===========================

A module which adds the :class:`.FrozenSettings` class.

Index
-----
.. currentmodule:: FOX.classes.frozen_settings
.. autosummary::
    FrozenSettings

API
---
.. autoclass:: FOX.classes.frozen_settings.FrozenSettings
    :members:
    :private-members:
    :special-members:

"""

from collections import abc
from types import MappingProxyType
from typing import (Any, Iterator, Optional, Callable, TypeVar, KeysView, ItemsView, ValuesView,
                    ClassVar, FrozenSet)

from assertionlib.dataclass import AbstractDataClass

__all__ = ['FrozenMapping']

KT = TypeVar('KT', bound=str)
KV = TypeVar('KV')


class FrozenMapping(AbstractDataClass, abc.Mapping):
    _PRIVATE_ATTR: ClassVar[FrozenSet[str]] = frozenset({'_view'})

    def __init__(self) -> None:
        super().__setattr__('_view', {})
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        """Implement :code:`setattr(self, name, value)`."""
        if name not in self._PRIVATE_ATTR:
            self._view[name] = value
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Implement :code:`delattr(self, name)`."""
        if name not in self._PRIVATE_ATTR:
            del self._view[name]
        super().__delattr__(name)

    @property
    def view(self):
        """Return a read-only view of all non-private instance attributes."""
        return MappingProxyType(self._view)

    @property
    def __bool__(self) -> Callable[[], bool]:
        """Get the :meth:`__bool__<types.MappingProxyType.__bool__>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.__bool__

    @property
    def __getitem__(self) -> Callable[[KT], KV]:
        """Get the :meth:`__getitem__<types.MappingProxyType.__getitem__>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.__getitem__

    @property
    def __iter__(self) -> Callable[[], Iterator[KT]]:
        """Get the :meth:`__iter__<types.MappingProxyType.__iter__>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.__iter__

    @property
    def __len__(self) -> Callable[[], int]:
        """Get the :meth:`__len__<types.MappingProxyType.__len__>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.__len__

    @property
    def __contains__(self) -> Callable[[KT], bool]:
        """Get the :meth:`__contains__<types.MappingProxyType.__contains__>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.__contains__

    @property
    def get(self) -> Callable[[KT, Optional[Any]], KV]:
        """Get the :meth:`get<types.MappingProxyType.get>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.get

    @property
    def keys(self) -> Callable[[], KeysView[KT]]:
        """Get the :meth:`keys<types.MappingProxyType.keys>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.keys

    @property
    def items(self) -> Callable[[], ItemsView[KT, KV]]:
        """Get the :meth:`items<types.MappingProxyType.items>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.items

    @property
    def values(self) -> Callable[[], ValuesView[KV]]:
        """Get the :meth:`values<types.MappingProxyType.values>` method of :attr:`FrozenMapping.view`."""  # noqa
        return self.view.values
