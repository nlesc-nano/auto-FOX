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
from typing import Sequence, Union, Type, Tuple, List, Dict, Iterable

import numpy as np
from pandas.core.generic import NDFrame

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable, Literal, TypedDict, SupportsIndex
else:
    from typing_extensions import Protocol, runtime_checkable, Literal, TypedDict

    @runtime_checkable
    class SupportsIndex(Protocol):
        """An :class:`~abc.ABC` with one abstract method :meth:`__index__`."""
        __slots__ = ()

        @abstractmethod
        def __index__(self) -> int:
            pass

__all__ = [
    'Scalar', 'ScalarType', 'ArrayLike', 'ArrayLikeOrScalar', 'ArrayOrScalar',
    'Literal', 'TypedDict'
]

#: Annotation for numerical scalars.
Scalar = Union[np.number, np.bool_, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Type[Scalar]

# ``_DtypeLikeNested`` and ``DtypeLike`` taken from numpy-stubs.
# Reference: https://github.com/numpy/numpy-stubs

# TODO: wait for support for recursive types
_DtypeLikeNested = Union[np.dtype, str, None, ScalarType]

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

    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    # TODO: use TypedDict when/if it's officially supported
    Dict[
        str,
        Union[
            Sequence[str],  # names
            Sequence[_DtypeLikeNested],  # formats
            Sequence[int],  # offsets
            Sequence[Union[bytes, str, None]],  # titles
            int,  # itemsize
        ],
    ],

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
