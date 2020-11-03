"""A module with type-hint related objects used throughout Auto-FOX.

Index
-----
.. currentmodule:: FOX.type_hints
.. autosummary::
{autosummary}

API
---
{autodata}

"""

from typing import Union, Type, TYPE_CHECKING

import numpy as np
from pandas.core.generic import NDFrame

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DtypeLike
else:
    from numpy import ndarray as ArrayLike, dtype as DtypeLike

__all__ = [
    'Scalar', 'ScalarType', 'ArrayLike', 'DtypeLike', 'ArrayLikeOrScalar', 'ArrayOrScalar'
]

#: Annotation for numerical scalars.
Scalar = Union[np.number, np.bool_, int, float, bool, complex]

#: Annotation for numerical scalar types.
ScalarType = Type[Scalar]

#: Annotation for array-like objects or numerical scalar.
ArrayLikeOrScalar = ArrayLike

#: Annotation for arrays.
Array = Union[np.ndarray, NDFrame]

#: Annotation for arrays or numerical scalars.
ArrayOrScalar = Union[Array, Scalar]

__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
