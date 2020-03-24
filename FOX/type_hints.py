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

import sys
from abc import abstractmethod
from typing import Sequence, TypeVar, Union, Type

import numpy as np
from pandas.core.generic import NDFrame

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable
else:
    from typing import Protocol, runtime_checkable

__all__ = [
    'Scalar', 'ScalarType', 'DtypeLike', 'ArrayLike',
    'ArrayLikeOrScalar', 'ArrayOrScalar', 'DtypeLike'
]

#: Annotation for numerical scalars.
Scalar = Union[np.generic, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Union[Type[np.generic], Type[int], Type[float], Type[bool], Type[complex]]

_DtypeLike = Union[None, str, ScalarType, np.dtype]
_DT_co = TypeVar('_DT_co', bound=_DtypeLike, covariant=True)


@runtime_checkable
class SupportsDtype(Protocol[_DT_co]):
    """An ABC with one abstract attribute :attr:`dtype`."""

    __slots__: tuple = ()

    @property
    @abstractmethod
    def dtype(self) -> _DT_co:
        pass


#: Annotation for numpy datatype-like objects.
DtypeLike = Union[_DtypeLike, SupportsDtype]


@runtime_checkable
class SupportsArray(Protocol):
    """An ABC with one abstract method :meth:`__array__`."""

    __slots__: tuple = ()

    @abstractmethod
    def __array__(self, dtype: DtypeLike = None) -> np.ndarray:
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
