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

from __future__ import annotations

import copy as pycopy
from typing import (Any, Sequence, Union)

from scm.plams import Settings

from ..functions.utils import append_docstring

__all__ = ['FrozenSettings']

Immutable = Union[str, int, float, tuple, frozenset, type]


@append_docstring(Settings)
class FrozenSettings(Settings):
    """An inmutable subclass of plams.Settings_.

    Has an additional attribute by the name of :attr:`FrozenSettings._hash`, which is used for
    caching the hash of this instance.
    The :meth:`FrozenSettings.__hash__` method will automatically create a new hash if
    :code:`bool(FrozenSettings._hash)` yields ``False``.

    .. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a :class:`FrozenSettings` instance."""
        dict.__init__(self, *args, **kwargs)

        # Fill the FrozenSettings instance by means of the dict.__setitem__ method
        for key, value in self.items():
            if isinstance(value, dict):
                Settings.__setitem__(self, key, FrozenSettings(value))
            elif isinstance(value, list):
                value = [FrozenSettings(i) if isinstance(i, dict) else i for i in value]
                Settings.__setitem__(self, key, value)

        # Cache the hash if this instance in self._hash
        dict.__setattr__(self, '_hash', 0)
        self.__hash__()

    def __missing__(self, key: Immutable) -> FrozenSettings:
        """Return a new (empty) :class:`FrozenSettings` instance."""
        return FrozenSettings()

    def __delitem__(self, key: Immutable) -> None:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError(f"'{self.__class__.__name__}' object does not support item deletion")

    def __setitem__(self, key: Immutable,
                    value: Any) -> None:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError(f"'{self.__class__.__name__}' object does not support item assignment")

    def __hash__(self) -> int:
        """Return the hash of this instance.

        The cached hash is automatically pulled from :attr:`FrozenSettings._hash` if available
        and is otherwise reconstructed, cached and returned.

        """
        # Retrieve the cached hash from self._hash
        ret = dict.__getattribute__(self, '_hash')
        if ret:
            return ret

        # Construct a new hash
        flat_dict = super().flatten()
        ret = 0
        for k, v in flat_dict.items():
            ret ^= hash(k + (v,))

        # Store the new hash in the cache and return it
        dict.__setattr__(self, '_hash', ret)
        return ret

    def copy(self, deep: bool = False) -> FrozenSettings:
        """Create a copy of this instance."""
        ret = FrozenSettings()
        copy_func = pycopy.deepcopy if deep else pycopy.copy

        # Copy items
        for key, value in self.items():
            if isinstance(value, dict):
                Settings.__setitem__(ret, key, value.copy())
            else:
                Settings.__setitem__(ret, key, copy_func(value))

        # Reconstruct the hash and store it in the cache
        dict.__setattr__(ret, '_hash', 0)
        ret.__hash__()
        return ret

    def __copy__(self) -> FrozenSettings:
        """Create a shallow copy of this instance"""
        return self.copy(deep=False)

    def __deepcopy__(self) -> FrozenSettings:
        """Create a deep copy of this instance"""
        return self.copy(deep=True)

    def set_nested(self, key_tuple: Sequence[Immutable],
                   value: Any, ignore_missing: bool = True) -> FrozenSettings:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError(f"'{self.__class__.__name__}' object does not support item assignment")

    @append_docstring(Settings.flatten)
    def flatten(self, flatten_list: bool = True) -> FrozenSettings:
        ret = super().flatten(flatten_list)
        return FrozenSettings(ret)

    @append_docstring(Settings.unflatten)
    def unflatten(self, unflatten_list: bool = True) -> FrozenSettings:
        ret = super().unflatten(unflatten_list)
        return FrozenSettings(ret)
