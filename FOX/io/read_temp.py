"""A module for reading and parsing temperatures.

Index
-----
.. currentmodule:: FOX.io.read_temp
.. autosummary::
    read_temperatures

API
---
.. autofunction:: read_temperatures

"""

import os
from itertools import islice
from typing import Union, Any

import numpy as np

__all__ = ['read_temperatures']


def read_temperatures(file: Union[str, bytes, os.PathLike], **kwargs: Any) -> np.ndarray:
    r"""Extract the temperatures from the passed ``cp2k-*.PARTICLES.temp`` file.

    Parameters
    ----------
    file : path_like
        A :term:`path-like object` pointing to the to-be read CP2K ``.temp`` file.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`open`.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}},)`
        A 1D array with the temperatures of all :math:`n_{\text{mol}}` molecules in **file**.

    """
    with open(file, 'r', **kwargs) as f:
        iterator = (i.split()[-1] for i in islice(f, 2, None, 2))
        return np.fromiter(iterator, dtype=np.float64)
