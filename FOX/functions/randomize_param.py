"""A module for randimizing parameters."""

from typing import Union
from collections import abc

import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(10))


def randomize(val: np.ndarray,
              val_max: Union[None, float, np.ndarray] = None,
              val_min: Union[None, float, np.ndarray] = None,
              n: int = 2) -> None:
    """pass

    Parameters
    ----------
    val : array-like [:class:`float`], shape :math:`(n,)`
        An array-like object containing the to-be randomized values.

    val_min : array-like [:class:`float`], shape :math:`(n,)`, optional
        An optional array-like with the new lower bounds of the to-be randomized values.
        Supplying a scalar will set the same lower bound for all values.

    val_min : array-like [:class:`float`], shape :math:`(n,)`, optional
        An optional array-like with the new upper bounds of the to-be randomized values.
        Supplying a scalar will set the same lower bound for all values.



    """
    val = np.asarray(val)

    if val_max is not None:  # Create an array with a upper bound for the new values
        if isinstance(val_max, abc.Iterable):
            val_max_ = np.array(val_max)
        else:
            val_max_ = np.full_like(val, val_max)
        posinf = np.isposinf(val_min)
        val_max_[posinf] = val[posinf] * n
        val_max_[~posinf] = np.minimum(val_max_[~posinf], val[~posinf] * n)
    else:
        val_max_ = val * n

    if val_min is not None:  # Create an array with a lower bound for the new values
        if isinstance(val_min, abc.Iterable):
            val_min_ = np.array(val_min)
        else:
            val_min_ = np.full_like(val, val_min)

        neginf = np.isneginf(val_min)
        val_min_[neginf] = val[neginf] / n
        val_min_[~neginf] = np.maximum(val_min_[~neginf], val[~neginf] / n)
    else:
        val_min_ = val / n

    # Construct new random values
    val_new = np.random.rand(len(val))
    val_new *= val_max_ - val_min_
    val_new += val_min_
    return val_new
