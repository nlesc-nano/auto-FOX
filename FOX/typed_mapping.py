"""
FOX.typed_mapping
=================

A module which adds the :class:`.TypedMapping` class.

Index
-----
.. currentmodule:: FOX.typed_mapping
.. autosummary::
    TypedMapping

API
---
.. autoclass:: TypedMapping
    :members:
    :private-members:
    :special-members:

"""

from collections import abc
from types import MappingProxyType
from typing import (Any, Iterator, Optional, Callable, TypeVar, KeysView, ItemsView, ValuesView,
                    ClassVar, FrozenSet)

from assertionlib.dataclass import AbstractDataClass

__all__ = ['TypedMapping']

KT = TypeVar('KT', bound=str)
KV = TypeVar('KV')


class TypedMapping(AbstractDataClass, abc.Mapping):
    """A :class:`Mapping<collections.abc.Mapping>` type which only a specific set of keys.

    Attributes
    ----------
    _ATTR : :class:`frozenset` [:class:`str`], classvar
        A frozenset containing all allowed keys.
        Should be defined at the class level.

    """

    _PRIVATE_ATTR: ClassVar[FrozenSet[str]] = frozenset({'_view', '_PRIVATE_ATTR'})
    _ATTR: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self) -> None:
        """Initialize a :class:`TypedMapping` instance."""
        super().__setattr__('_view', {})  # Dict[str, Any]
        super().__init__()

        cls = type(self)
        for k in cls._ATTR:
            super().__setattr__(k, None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Implement :code:`setattr(self, name, value)`."""
        # These values should be mutable
        if name in self._PRIVATE_ATTR:
            super().__setattr__(name, value)
            return

        # These values can only be changed once; i.e. when they're changed from None to Any
        elif name in self._ATTR and getattr(self, name, None) is None:
            self._view[name] = value
            super().__setattr__(name, value)
            return

        # Uhoh, trying to mutate something which should not be changed
        cls_name = self.__class__.__name__
        if name in self._ATTR and getattr(self, name, None) is not None:
            raise AttributeError(f"{cls_name!r} object attribute {name!r} is read-only")
        else:
            raise AttributeError(f"{cls_name!r} object has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        """Implement :code:`delattr(self, name)`."""
        raise AttributeError(f"{self.__class__.__name__!r} object attribute {name!r} is read-only")

    def __setitem__(self, name: str, value: Any) -> None:
        """Implement :code:`self[name] = value`."""
        if name in self._PRIVATE_ATTR:
            cls_name = self.__class__.__name__
            raise AttributeError(f"{cls_name!r} object attribute {name!r} is read-only")
        self.__setattr__(name, value)

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
