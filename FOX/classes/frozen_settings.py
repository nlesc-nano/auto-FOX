"""A module which adds the :class:`.FrozenSettings` class."""

from __future__ import annotations

from typing import (Any, Hashable, Sequence)

from scm.plams import Settings

from FOX.functions.utils import append_docstring

__all__ = ['FrozenSettings']


@append_docstring(Settings)
class FrozenSettings(Settings):
    """An inmutable subclass of plams.Settings_.

    .. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
    """

    def __init__(self, *arg, **kwarg) -> None:
        """Construct a :class:`FrozenSettings` instance."""
        dict.__init__(self, *arg, **kwarg)

        # Fill the FrozenSettings instance by means of the dict.__setitem__ method
        for key, value in self.items():
            if isinstance(value, dict):
                Settings.__setitem__(self, key, FrozenSettings(value))
            elif isinstance(value, list):
                value = [FrozenSettings(i) if isinstance(i, dict) else i for i in value]
                Settings.__setitem__(self, key, value)

    def __missing__(self, name: Hashable) -> FrozenSettings:
        """Return a new (empty) :class:`FrozenSettings` instance."""
        return FrozenSettings()

    def __delitem__(self, name: Hashable) -> None:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError("'FrozenSettings' object does not support item deletion")

    def __setitem__(self, name: Hashable,
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

    def set_nested(self, key_tuple: Sequence[Hashable],
                   value: Any,
                   ignore_missing: bool = True) -> FrozenSettings:
        """Raise a :exc:`TypeError`, :class:`FrozenSettings` instances are immutable."""
        raise TypeError("'FrozenSettings' object does not support item assignment")

    @append_docstring(Settings.flatten)
    def flatten(self, flatten_list=True) -> FrozenSettings:
        """"""
        ret = super().flatten(flatten_list)
        return FrozenSettings(ret)

    @append_docstring(Settings.unflatten)
    def unflatten(self, unflatten_list=True) -> FrozenSettings:
        """"""
        ret = super().unflatten(unflatten_list)
        return FrozenSettings(ret)
