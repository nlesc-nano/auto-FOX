"""A module which adds the :class:`TypedMapping` class.

Index
-----
.. currentmodule:: FOX.typed_mapping
.. autosummary::
    TypedMapping
    TypedMapping.__setattr__
    TypedMapping.__delattr__
    TypedMapping.__setitem__
    TypedMapping.__bool__
    TypedMapping.__getitem__
    TypedMapping.__iter__
    TypedMapping.__len__
    TypedMapping.__contains__
    TypedMapping.get
    TypedMapping.keys
    TypedMapping.items
    TypedMapping.values

API
---
.. autoclass:: TypedMapping
.. automethod:: TypedMapping.__setattr__
.. automethod:: TypedMapping.__delattr__
.. automethod:: TypedMapping.__setitem__
.. automethod:: TypedMapping.__bool__
.. automethod:: TypedMapping.__getitem__
.. automethod:: TypedMapping.__iter__
.. automethod:: TypedMapping.__len__
.. automethod:: TypedMapping.__contains__
.. automethod:: TypedMapping.get
.. automethod:: TypedMapping.keys
.. automethod:: TypedMapping.items
.. automethod:: TypedMapping.values

"""

from collections import abc
from types import MappingProxyType
from typing import (Any, Iterator, Optional, Callable, TypeVar, KeysView, ItemsView, ValuesView,
                    ClassVar, FrozenSet, NoReturn, Mapping)

from assertionlib.dataclass import AbstractDataClass

__all__ = ['TypedMapping']

KT = TypeVar('KT', bound=str)
KV = TypeVar('KV')


class TypedMapping(AbstractDataClass, abc.Mapping):
    """A :class:`Mapping<collections.abc.Mapping>` type which only allows a specific set of keys.

    Values cannot be altered after their assignment.

    Attributes
    ----------
    _PRIVATE_ATTR : :class:`frozenset` [:class:`str`], classvar
        A frozenset defining all private instance variables.

    _ATTR : :class:`frozenset` [:class:`str`], classvar
        A frozenset containing all allowed keys.
        Should be defined at the class level.

    view : :class:`MappingProxyType<types.MappingProxyType>` [:class:`str`, :data:`Any<typing.Any>`]
        Return a read-only view of all items specified in :attr:`TypedMapping._ATTR`.

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

    @property
    def view(self) -> Mapping[KT, KV]:
        """Return a read-only view of all items specified in :attr:`TypedMapping._ATTR`."""
        return MappingProxyType(self._view)

    def __setattr__(self, name: str, value: Any) -> None:
        """Implement :code:`setattr(self, name, value)`.

        Attributes specified in :attr:`TypedMapping._PRIVATE_ATTR` can freely modified.
        Attributes specified in :attr:`TypedMapping._ATTR` can only be modified when the
        previous value is ``None``.
        All other attribute cannot be modified any further.

        """
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

    def __delattr__(self, name: str) -> NoReturn:
        """Implement :code:`delattr(self, name)`.

        Raises an :exc:`AttributeError`, instance variables cannot be deleted.

        """
        raise AttributeError(f"{self.__class__.__name__!r} object attribute {name!r} is read-only")

    def __setitem__(self, name: str, value: Any) -> None:
        """Implement :code:`self[name] = value`.

        Serves as an alias for :meth:`TypedMapping.__setattr__` when **name** is in
        :meth:`TypedMapping._ATTR`.

        """
        if name in self._PRIVATE_ATTR:
            raise KeyError(repr(name))
        self.__setattr__(name, value)

    @property
    def __bool__(self) -> Callable[[], bool]:
        """Get the :meth:`__bool__()<types.MappingProxyType.__bool__>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.__bool__

    @property
    def __getitem__(self) -> Callable[[KT], KV]:
        """Get the :meth:`__getitem__()<types.MappingProxyType.__getitem__>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.__getitem__

    @property
    def __iter__(self) -> Callable[[], Iterator[KT]]:
        """Get the :meth:`__iter__()<types.MappingProxyType.__iter__>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.__iter__

    @property
    def __len__(self) -> Callable[[], int]:
        """Get the :meth:`__len__()<types.MappingProxyType.__len__>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.__len__

    @property
    def __contains__(self) -> Callable[[KT], bool]:
        """Get the :meth:`__contains__()<types.MappingProxyType.__contains__>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.__contains__

    @property
    def get(self) -> Callable[[KT, Optional[Any]], KV]:
        """Get the :meth:`get()<types.MappingProxyType.get>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.get

    @property
    def keys(self) -> Callable[[], KeysView[KT]]:
        """Get the :meth:`keys()<types.MappingProxyType.keys>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.keys

    @property
    def items(self) -> Callable[[], ItemsView[KT, KV]]:
        """Get the :meth:`items()<types.MappingProxyType.items>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.items

    @property
    def values(self) -> Callable[[], ValuesView[KV]]:
        """Get the :meth:`values()<types.MappingProxyType.values>` method of :attr:`TypedMapping.view`."""  # noqa
        return self.view.values
