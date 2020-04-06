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
Scalar = Union[np.generic, int, float, bool]

#: Annotation for numerical scalar types.
ScalarType = Union[Type[np.generic], Type[int], Type[float], Type[bool]]

_DtypeLike = Union[None, str, ScalarType, np.dtype]
_DT_co1 = TypeVar('_DT_co1', bound=_DtypeLike, covariant=True)

_KT_co = TypeVar('_KT_co', bound=Hashable, covariant=True)
_VT_co = TypeVar('_VT_co', covariant=True)
_ST = TypeVar('_ST', bound=Scalar)
_ST_co = TypeVar('_ST_co', bound=Scalar, covariant=True)


@runtime_checkable
class SupportsDtype(Protocol[_DT_co1]):
    """An ABC with one abstract attribute :attr:`dtype`."""

    __slots__: Tuple[str, ...] = ()

    @property
    @abstractmethod
    def dtype(self) -> _DT_co1:
        pass


#: Annotation for numpy datatype-like objects.
DtypeLike = Union[_DtypeLike, SupportsDtype]
_DT_co2 = TypeVar('_DT_co2', bound=DtypeLike, covariant=True)


class NDArray(np.ndarray, Sequence[_ST], Generic[_ST]):
    """A (barebones) generic version of :class:`numpy.ndarray`."""

    def __array__(self, dtype: DtypeLike = None) -> NDArray[_ST]:
        pass


@runtime_checkable
class SupportsArray(Protocol[_ST_co]):
    """An ABC with one abstract method :attr:`__array__`."""

    __slots__: Tuple[str, ...] = ()

    @overload
    @abstractmethod
    def __array__(self, dtype: _ST) -> NDArray[_ST]: ...

    @overload
    @abstractmethod
    def __array__(self, dtype: DtypeLike) -> NDArray[_ST]: ...

    @abstractmethod
    def __array__(self, dtype=None):
        pass


#: Annotation for array-like objects.
ArrayLike = Union[Sequence, SupportsArray]

#: Annotation for array-like objects or numerical scalar.
ArrayLikeOrScalar = Union[ArrayLike, Scalar]

#: Annotation for arrays or numerical scalars.
ArrayOrScalar = Union[np.ndarray, NDFrame, Scalar]


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
