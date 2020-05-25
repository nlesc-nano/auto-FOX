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
.. autoattribute:: TypedMapping.__bool__
.. autoattribute:: TypedMapping.__getitem__
.. autoattribute:: TypedMapping.__iter__
.. autoattribute:: TypedMapping.__len__
.. autoattribute:: TypedMapping.__contains__
.. autoattribute:: TypedMapping.get
.. autoattribute:: TypedMapping.keys
.. autoattribute:: TypedMapping.items
.. autoattribute:: TypedMapping.values

"""

import sys
from types import MappingProxyType
from typing import NoReturn, Mapping, Generic, ClassVar, FrozenSet, TypeVar

__all__ = ['TypedMapping']

KT = TypeVar('KT', bound=str)
VT = TypeVar('VT')


class TypedMapping(Mapping[KT, VT], Generic[KT, VT]):
    """A :class:`Mapping<collections.abc.Mapping>` type which only allows a specific set of keys.

    Values cannot be altered after their assignment.

    """

    __slots__ = ('__weakref__', '_view')
    KEYS: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a :class:`TypedMapping` instance."""
        super().__init__()

        cls = type(self)
        dct = dict(*args, **kwargs)
        if not dct.keys() == cls.KEYS:
            diff = dct.keys() - cls.KEYS
            raise TypeError(f"Invalid keys: {diff}")

        object.__setattr__(self, '_view', MappingProxyType(dct))  # Dict[KT, VT]

    def __reduce__(self):
        """Helper function for :mod:`pickle`."""
        return type(self), (self._view.copy(),)

    def __getattr__(self, name):
        """Implement :func:`getattr(self, name)<getattr>`."""
        try:
            return self[name]
        except KeyError:
            return getattr(self, name)

    def __setattr__(self, name, value) -> NoReturn:  # Attributes are read-only
        """Implement :func:`setattr(self, name, value)<setattr>`."""
        raise self._attributeError(name)

    def __delattr__(self, name) -> NoReturn:  # Attributes are read-only
        """Implement :func:`delattr(self, name)<delattr>`."""
        raise self._attributeError(name)

    def _attributeError(self, name) -> AttributeError:
        """Return an :exc:`AttributeError`; attributes of this instance are read-only."""
        cls_name = self.__class__.__name__
        if hasattr(self, name):
            return AttributeError(f"attribute {name!r} of {cls_name!r} objects is not writable")
        else:
            return AttributeError(f"{cls_name!r} object has no attribute {name!r}")

    def copy(self):
        """:class:`TypedMapping` objects are immutable; return :code:`self`."""
        return self

    def __copy__(self):
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self, memo=None):
        """Implement :func:`copy.deepcopy(self, memo=...)<copy.deepcopy>`."""
        return self

    @property
    def __eq__(self):
        """Implement :func:`self == value<object.__eq__>`."""
        return self._view.__eq__

    @property
    def __bool__(self):
        """Implement :func:`bool(value)<object.__bool__>`."""
        return self._view.__bool__

    @property
    def __getitem__(self):
        """Implement :func:`self[key]<dict.__getitem__>`."""
        return self._view.__getitem__

    @property
    def __iter__(self):
        """Implement :func:`iter(self)<iter>`."""
        return self._view.__iter__

    @property
    def __len__(self):
        """Implement :func:`len(self)<len>`."""
        return self._view.__len__

    @property
    def __contains__(self):
        """Implement :func:`key in self<dict.__contains__>`."""
        return self._view.__contains__

    @property
    def get(self):
        """Implement :func:`self.get(key, default=...)<dict.get>`."""
        return self._view.get

    @property
    def keys(self):
        """Return a set-like object providing a view on the keys in :code:`self`."""
        return self._view.keys

    @property
    def items(self):
        """Return a set-like object providing a view on the key/value pairs in :code:`self`."""
        return self._view.items

    @property
    def values(self):
        """Return an object providing a view on the values in :code:`self`."""
        return self._view.values

    if sys.version_info >= (3, 8):
        @property
        def __reversed__(self):
            """Implement :func:`reversed(self)<reversed>`; requires python 3.8 or later."""
            return self._view.__reversed__

    if sys.version_info >= (3, 9):
        @property
        def __or__(self):
            """Implement :func:`self | value<object.__or__>`."""
            return self._view.__or__
