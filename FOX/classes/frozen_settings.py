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

from typing import (Any, Sequence, Union)

from scm.plams import Settings

from FOX.functions.utils import append_docstring

__all__ = ['FrozenSettings']

Immutable = Union[str, int, float, tuple, frozenset]


@append_docstring(Settings)
class FrozenSettings(Settings):
    """An inmutable subclass of plams.Settings_.

    .. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
    """

    def __init__(self, *args, **kwargs) -> None:
        """Construct a :class:`FrozenSettings` instance."""
        dict.__init__(*args, **kwargs)

        # Fill the FrozenSettings instance by means of the dict.__setitem__ method
        for key, value in self.items():
            if isinstance(value, dict):
                Settings.__setitem__(self, key, FrozenSettings(value))
            elif isinstance(value, list):
                value = [FrozenSettings(i) if isinstance(i, dict) else i for i in value]
                Settings.__setitem__(self, key, value)

    def __missing__(self, key: Immutable) -> FrozenSettings:
        """Return a new (empty) :class:`FrozenSettings` instance."""
        return FrozenSettings()

    def __delitem__(self, key: Immutable) -> None:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError("'FrozenSettings' object does not support item deletion")

    def __setitem__(self, key: Immutable,
                    value: Any) -> None:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError("'FrozenSettings' object does not support item assignment")

    def __copy__(self) -> FrozenSettings:
        """Create a copy of this instance."""
        ret = FrozenSettings()
        for key, value in self.items():
            if isinstance(value, FrozenSettings):
                Settings.__setitem__(ret, key, value.copy())
            else:
                Settings.__setitem__(ret, key, value)
        return ret

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        flat_dict = super().flatten()
        ret = 0
        for k, v in flat_dict.items():
            ret ^= hash(k + (v,))
        return ret

    def __str__(self) -> str:
        def _str(dict_: dict, indent: int) -> str:
            ret = ''
            for key, value in dict_.items():
                ret += ' ' * indent + str(key) + ': \t'
                indent_new = indent + 1 + len(str(key))
                if isinstance(value, dict):
                    ret += '\n' + value._str(indent_new)
                else:
                    ret += _indent(value, indent_new) + '\n'
            return ret

        def _indent(value: object, indent: int) -> str:
            str_list = str(value).split('\n')
            joiner = '\n' + indent * ' '
            return joiner.join(i for i in str_list)

        return _str(self, 0)

    def __repr__(self) -> str:
        return self.__str__()

    def set_nested(self, key_tuple: Sequence[Immutable],
                   value: Any, ignore_missing: bool = True) -> FrozenSettings:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError("'FrozenSettings' object does not support item assignment")

    @append_docstring(Settings.flatten)
    def flatten(self, flatten_list: bool = True) -> FrozenSettings:
        """"""
        ret = super().flatten(flatten_list)
        return FrozenSettings(ret)

    @append_docstring(Settings.unflatten)
    def unflatten(self, unflatten_list: bool = True) -> FrozenSettings:
        """"""
        ret = super().unflatten(unflatten_list)
        return FrozenSettings(ret)
