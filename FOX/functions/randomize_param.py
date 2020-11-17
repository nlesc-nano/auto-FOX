"""A module for randimizing parameters.

Index
-----
.. currentmodule:: FOX.functions.randomize_param
.. autosummary::
    randomize

API
---
.. autofunction:: randomize

"""

from typing import Callable, Optional

import numpy as np

__all__ = ['randomize']


def randomize(val: np.ndarray,
              val_max: Optional[np.ndarray] = None,
              val_min: Optional[np.ndarray] = None,
              n: float = 2.0) -> None:
    """Randomize the values in **val**.

    Parameters
    ----------
    val : array-like [:class:`float`], shape :math:`(m,)`
        An array-like object containing the to-be randomized values.

    val_min : array-like [:class:`float`], shape :math:`(m,)`, optional
        An optional array-like with the new lower bounds of the to-be randomized values.
        Supplying a scalar will set the same lower bound for all values.

    val_min : array-like [:class:`float`], shape :math:`(m,)`, optional
        An optional array-like with the new upper bounds of the to-be randomized values.
        Supplying a scalar will set the same lower bound for all values.

    n : :class:`float`
        Ensure that the new parameters cannot be higher or lower than, respectivelly,
        :math:`/text{val} * n` and :math:`/text{val} / n`.
        If both :math:`n` and the extremes are specified,
        take whichever value is high/lower

    Returns
    -------
    :class:`np.ndarray` [:class:`float`], shape :math:`(m,)`
        A new array of randomized values.

    """
    val = np.asarray(val)

    # Parse the extremites
    val_max_ = _parse_extremite(val, val_max, n, or_func=np.minimum)
    val_min_ = _parse_extremite(val, val_max, 1/n, or_func=np.maximum)

    # Construct new random values
    val_new = np.random.rand(len(val))
    val_new *= val_max_ - val_min_
    val_new += val_min_
    return val_new


#: Functions for creating new arrays from the elements of the two input arrays.
OrFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _parse_extremite(val: np.ndarray, val_extreme: Optional[np.ndarray],
                     n: float = 2.0, or_func: OrFunc = np.minimum) -> np.ndarray:
    """Parse the **val_min** and **val_max** parameters of :func:`randomize`."""
    if val_extreme is None:
        return val * n

    ret = np.array(val_extreme)
    isinf = np.isinf(ret)
    if ret.size == 1:
        return or_func(val * n, ret)

    ret[isinf] = val[isinf] * n
    ret[~isinf] = or_func(ret[~isinf], val[~isinf] * n)
