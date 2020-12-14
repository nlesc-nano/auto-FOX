"""A module for reading and parsing volumes.

Index
-----
.. currentmodule:: FOX.io.read_volume
.. autosummary::
    read_volumes

API
---
.. autofunction:: read_volumes

"""

import os
from itertools import islice
from typing import Union, Any

import numpy as np
from scm.plams import Units

__all__ = ['read_volumes']


def read_volumes(
    file: Union[str, bytes, os.PathLike], unit: str = 'angstrom', **kwargs: Any
) -> np.ndarray:
    r"""Extract the cell volumes from the passed ``cp2k-*.cell`` file.

    Parameters
    ----------
    file : path_like
        A :term:`path-like object` pointing to the to-be read CP2K ``.cell`` file.
    unit : :class:`str`
        The unit in which the to-be returned cell volume should be expressed
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`open`.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}},)`
        A 1D array with the volumes of all :math:`n_{\text{mol}}` molecules in **file**.

    """
    with open(file, 'r', **kwargs) as f:
        iterator = (i.split()[-1] for i in islice(f, 1, None))
        ret = np.fromiter(iterator, dtype=np.float64)

    if unit != 'angstrom':
        ret *= Units.conversion_ratio('angstrom', unit)**3
    return ret
