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
from typing import Sequence, Union, Type, Tuple, Any, List, Dict, Iterable

import numpy as np
from pandas.core.generic import NDFrame

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable, Literal, TypedDict
else:
    from typing import Protocol, runtime_checkable, Literal, TypedDict

__all__ = [
    'Scalar', 'ScalarType', 'ArrayLike', 'ArrayLikeOrScalar', 'ArrayOrScalar',
    'Literal', 'TypedDict'
]

#: Annotation for numerical scalars.
Scalar = Union[np.generic, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Union[Type[np.generic], Type[int], Type[float], Type[bool], Type[complex]]

# ``_DtypeLikeNested`` and ``DtypeLike`` taken from numpy-stubs.
# Reference: https://github.com/numpy/numpy-stubs

_DtypeLikeNested = Any  # TODO: wait for support for recursive types


class DtypeDict(TypedDict):
    """A :class:`~typing.TypedDict` representing one of the :class:`numpy.dtype` inputs."""

    names: Sequence[str]
    formats: Sequence[_DtypeLikeNested]
    offsets: Sequence[int]
    titles: Sequence[Union[bytes, str, None]]
    itemsize: int


# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DtypeLike = Union[
    np.dtype,

    # default data type (float64)
    None,

    # array-scalar types and generic types
    ScalarType,  # TODO: enumerate these when we add type hints for numpy scalars

    # TODO: add a protocol for anything with a dtype attribute
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,

    # (flexible_dtype, itemsize)
    Tuple[_DtypeLikeNested, int],

    # (fixed_dtype, shape)
    Tuple[_DtypeLikeNested, Union[int, Sequence[int]]],

    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    List[Tuple[str, _DtypeLikeNested, Union[int, Sequence[int]]]],

    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}
    DtypeDict,

    # {'field1': ..., 'field2': ..., ...}
    Dict[str, Tuple[_DtypeLikeNested, int]],

    # (base_dtype, new_dtype)
    Tuple[_DtypeLikeNested, _DtypeLikeNested],
]


@runtime_checkable
class SupportsArray(Protocol):
    """An ABC with one abstract method :meth:`__array__`."""

    __slots__: Union[str, Iterable[str]] = ()

    @abstractmethod
    def __array__(self, dtype: DtypeLike = None) -> np.ndarray:
        pass


#: Annotation for array-like objects.
ArrayLike = Union[Sequence[Scalar], SupportsArray, np.ndarray]

#: Annotation for array-like objects or numerical scalar.
ArrayLikeOrScalar = Union[ArrayLike, Scalar]

#: Annotation for arrays.
Array = Union[np.ndarray, NDFrame]

#: Annotation for arrays or numerical scalars.
ArrayOrScalar = Union[Array, Scalar]


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
