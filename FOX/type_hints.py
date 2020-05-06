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
from typing import Sequence, Union, Type, Tuple, List, Dict, Iterable, Any, TYPE_CHECKING

import numpy as np
from pandas.core.generic import NDFrame

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

if TYPE_CHECKING:
    from numpy import DtypeLike, SupportsArray
else:
    DtypeLike = Any
    SupportsArray = Any

__all__ = [
    'Scalar', 'ScalarType', 'ArrayLike', 'ArrayLikeOrScalar', 'ArrayOrScalar',
    'Literal', 'TypedDict'
]

#: Annotation for numerical scalars.
Scalar = Union[np.number, np.bool_, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Type[Scalar]

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
