"""
FOX.classes.multi_mol_magic
===========================

A Module for setting up the magic methods and properties of the :class:`.MultiMolecule` class.

Index
-----
.. currentmodule:: FOX.classes.multi_mol_magic
.. autosummary::
    _MultiMolecule

API
---
.. autoclass:: FOX.classes.multi_mol_magic._MultiMolecule
    :members:
    :private-members:
    :special-members:

"""

import reprlib
import textwrap
from typing import (Any, Iterable)

import numpy as np

__all__ = ['MultiMolRepr']


class MultiMolRepr(reprlib.Repr):
    """A :class:`reprlib.Repr` subclass for creating :class:`.MultiMolecule` strings."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.maxdict = 6
        self.maxndarray = 4
        self._ndformatter = {'str': '{:}'.format, 'int': '{:4d}'.format, 'float': '{:8.4f}'.format}

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        attr_dict = vars(self)
        width = max(len(k) for k in attr_dict)
        iterator = sorted(attr_dict.items(), key=str)

        ret = ',\n'.join(f'{k:{width}} = {repr(v)}' for k, v in iterator if k[0] != '_')
        _ret = textwrap.indent(ret, 4 * ' ')
        return f'{self.__class__.__name__}(\n{_ret}\n)'

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        return self.__str__()

    def __hash__(self) -> int:
        """Return the hash of this instance."""
        ret = 0
        for k, v in vars(self).items():
            if k == '_ndformatter':
                continue
            ret ^= hash((k, v))
        return ret

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if self.__class__ is not value.__class__:
            return False

        # Check if the object attribute values are identical
        try:
            for k, v1 in vars(self).items():
                v2 = getattr(value, k)
                assert v1 == v2
        except (AttributeError, AssertionError):
            return False  # An attribute is missing or not equivalent

        return True

    def repr1(self, x: Any, level: int) -> str:
        if isinstance(x, np.ndarray):
            return self.repr_ndarray(x, level)
        elif isinstance(x, dict):
            return self.repr_dict(x, level)
        else:
            return super().repr1(x, level)

    def repr_ndarray(self, obj: np.ndarray, level: int) -> str:
        """Convert an array into a string"""
        edgeitems = self.maxndarray // 2
        with np.printoptions(threshold=1, edgeitems=edgeitems, formatter=self._ndformatter):
            return np.ndarray.__str__(obj)

    def _repr_iterable(self, obj: Iterable, level: int, left: str,
                       right: str, maxiter: int, trail: str = '') -> str:
        ar = np.array(obj)
        if ar.ndim != 1 or ar.dtype is np.dtype(object):
            return super()._repr_iterable(obj, level, left, right, maxiter, trail)

        edgeitems = maxiter // 2
        with np.printoptions(threshold=1, edgeitems=edgeitems, formatter=self._ndformatter):
            ret = repr(ar).strip('array([').rstrip('])')
            return f'{left}{ret}{right}'

    def repr_dict(self, obj: dict, level: int) -> str:
        """Convert a dictionary into a string."""

        def _repr_dict(k: Any, v: Any, offset: int, level: int) -> str:
            """Create key/value pairs for :meth:`MultiMolRepr.repr_dict`."""
            key = self.repr1(k, level)
            value = self.repr1(v, level)
            return f'{key:{offset}}' + f': {value}'

        n = len(obj)
        if n == 0:
            return '{}'
        elif level <= 0:
            return '{...}'
        elif level != self.maxlevel:
            return super().repr_dict(obj, level)

        lvl = level - 1
        offset = 1 + max(len(repr(k)) for k in obj)
        pieces = [_repr_dict(k, v, offset, lvl) for k, v in sorted(obj.items(), key=str)]

        if len(pieces) > self.maxdict:
            i = self.maxdict // 2
            ret = ',\n '.join(pieces[:i])
            ret += ',\n ...,\n '
            ret += ',\n '.join(pieces[-i:])
        else:
            ret = ',\n '.join(item for item in pieces)

        return '{' + ret + '}'
