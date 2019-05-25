"""A module which adds the :class:`.FrozenSettings` class."""

from __future__ import annotations

from typing import (Callable, Hashable, Any, Iterable)

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
            elif isinstance(value, (list, tuple)):
                value = value.__class__(
                    FrozenSettings(i) if isinstance(i, dict) else i for i in value
                )
                Settings.__setitem__(self, key, value)

    def get_nested_value(self, key: Iterable[Hashable]) -> Any:
        """Return the value from a nested dictionary associated with all keys in **key**.

        Example:

        .. code:: python

            >>> dict_ = {'a': {'b': {'c': 'value1'}}}
            >>> key = ('a', 'b', 'c')
            >>> get_nested(dict_, key)
            'value1'

            >>> dict_ = {'a': {'b': [['value2']]}}
            >>> key = ('a', 'b', 0, 0)
            >>> get_nested(dict_, key)
            'value2'

        Parameters:
            Iterable[Hashable]:
                An iterable with (nested) keys beloning to this :class:`FrozenSettings` instance.

        Returns:
            object:
                The value (nested) associated the keys in **key**.
        """
        def iterate(dict_, i):
            return value[i]

        value = self
        for i in key:
            value = iterate(value, i)
        return value

    def __missing__(self, name: Hashable) -> FrozenSettings:
        """Return a new (empty) :class:`FrozenSettings` instance."""
        return FrozenSettings()

    def __delitem__(self, name: Hashable) -> None:
        """Raise a TypeError."""
        raise TypeError("'FrozenSettings' object does not support item deletion")

    def __setitem__(self, name: Hashable,
                    value: Any) -> None:
        """Raise a TypeError."""
        raise TypeError("'FrozenSettings' object does not support item assignment")

    def __copy__(self) -> FrozenSettings:
        """Create a copy of **self**."""
        ret = FrozenSettings()
        for key, value in self.items():
            if isinstance(self[key], FrozenSettings):
                Settings.__setitem__(ret, key, value.copy())
            else:
                Settings.__setitem__(ret, key, value)
        return ret

    def __hash__(self) -> int:
        """Return the hash of self.

        Hashes are constructed from tuples containing all keys associated with a specific value,
        in addition to containing the value itself.
        """
        ret = 0
        flat = self.flatten()
        for key, value in flat.items():
            ret ^= hash(key + (value,))
        return ret

    def flatten(self) -> FrozenSettings:
        """Flatten a nested dictionary.

        The keys of the to be returned dictionary consist are tuples with the old (nested) keys
        of **input_dict**.

        .. code-block:: python

            >>> print(input_dict)
            {'a': {'b': {'c': True}}}

            >>> output_dict = flatten_dict(input_dict)
            >>> print(output_dict)
            {('a', 'b', 'c'): True}

        Returns:
            |FOX.FrozenSettings|_ (keys: |tuple|_):
                A newly flattened :class:`FrozenSettings` instance.
        """
        def concatenate_dict(key_prepend: tuple, dict_: dict) -> None:
            for key, value in dict_.items():
                key = key_prepend + (key, )
                if isinstance(value, dict):
                    concatenate_dict(key, value)
                else:
                    Settings.__setitem__(ret, key, value)

        # Changes keys into tuples
        ret = FrozenSettings()
        concatenate_dict((), self)
        return ret

    @append_docstring(Settings.as_dict)
    def as_dict(self, ret_type: Callable = dict) -> dict:
        """Convert a :class:`.FrozenSettings` instance into a dictionary."""
        ret = ret_type.__class__()
        for key, value in self.items():
            if isinstance(value, Settings):
                ret[key] = value.as_dict(ret_type)
            elif isinstance(value, (list, tuple)):
                ret[key] = value.__class__(
                    i.as_dict() if isinstance(i, FrozenSettings) else i for i in value
                )
            else:
                ret[key] = value

        return ret
