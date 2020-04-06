"""
FOX.type_hints
==============

A module with type-hint related objects used throughout Auto-FOX.

Index
-----
.. currentmodule:: FOX.type_hints
.. autosummary::
{autosummary}

API
---
{autodata}

"""

from __future__ import annotations

import sys
from abc import abstractmethod
from typing import Sequence, TypeVar, Union, Type, Generic, Hashable, Tuple, overload

import numpy as np
from pandas.core.generic import NDFrame

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable, Literal, TypedDict
else:
    from typing import Protocol, runtime_checkable, Literal, TypedDict

__all__ = [
    'Scalar', 'ScalarType', 'ArrayLike',
    'ArrayLikeOrScalar', 'ArrayOrScalar',
    'Literal', 'NDArray', 'TypedDict'
]

#: Annotation for numerical scalars.
Scalar = Union[np.generic, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Union[Type[np.generic], Type[int], Type[float], Type[bool], Type[complex]]

_DtypeLike = Union[None, str, ScalarType, np.dtype]


@runtime_checkable
class SupportsDtype(Protocol):
    """An ABC with one abstract attribute :attr:`dtype`."""

    __slots__: Tuple[str, ...] = ()

    @property
    @abstractmethod
    def dtype(self) -> _DtypeLike:
        pass


#: Annotation for numpy datatype-like objects.
DtypeLike = Union[_DtypeLike, SupportsDtype]


@runtime_checkable
class SupportsArray(Protocol):
    """An ABC with one abstract method :meth:`__array__`."""

    __slots__: Tuple[str, ...] = ()

    @abstractmethod
    def __array__(self, dtype: DtypeLike = None) -> np.ndarray:
        pass


#: Annotation for array-like objects.
ArrayLike = Union[Sequence, SupportsArray]

#: Annotation for array-like objects or numerical scalar.
ArrayLikeOrScalar = Union[ArrayLike, Scalar]

#: Annotation for arrays or numerical scalars.
Array = Union[np.ndarray, NDFrame]

#: Annotation for arrays or numerical scalars.
ArrayOrScalar = Union[Array, Scalar]


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
