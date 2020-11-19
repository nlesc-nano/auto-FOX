"""A module for potential energy surfaces from KF binary files.

Index
-----
.. currentmodule:: FOX.io.read_kf
.. autosummary::
    read_kf

API
---
.. autofunction:: read_kf

"""

from os import PathLike
from itertools import chain
from typing import Tuple, Dict, List, Union

import numpy as np

from scm.plams import KFReader
from nanoutils import group_by_values

__all__ = ['read_kf']

IdxDict = Dict[str, List[int]]


def read_kf(filename: Union[str, 'PathLike[str]']) -> Tuple[np.ndarray, IdxDict]:
    """Read a KF binary file containing a potential energy surface.

    Returns the following items:

        * An array with the cartesian coordinates of :math:`m` molecules
          consisting of :math:`n` atoms.

        * A dictionary with atomic symbols and lists of matching atomic indices.

    Current KF file support is limited to those produced with the AMS_ engine.

    .. _ams: https://www.scm.com/product/ams/

    Parameters
    ----------
    filename : str
        The path+filename of the KF binary file.

    Returns
    -------
    :math:`m*n*3` |np.ndarray|_ [|np.float64|_] and |dict|_ [|str|_, |list|_ [|int|_]]:
        A 3D array with cartesian coordinates and a dictionary
        with atomic symbols as keys and lists of matching atomic indices as values.

    """
    kf = KFReader(filename)
    atom_count = kf.read('Molecule', 'nAtoms')
    mol_count = kf.read('History', 'nEntries')

    # Create an empty m*(n*3) xyz array
    shape = mol_count, atom_count, 3
    count = mol_count * atom_count * 3

    # Fill the xyz array with cartesian coordinates
    mol_range = range(1, 1+mol_count)
    iterator = chain.from_iterable(kf.read('History', f'Coords({i})') for i in mol_range)
    xyz = np.fromiter(iterator, dtype=float, count=count)
    xyz.shape = shape

    return xyz, _get_idx_dict(kf)


def _get_idx_dict(kf: KFReader) -> IdxDict:
    """Extract atomic symbols and matching atomic indices from **kf**.

    Parameters
    ----------
    kf : |plams.KFReader|_
        A KFReader instance constructed from a KF binary file.

    Returns
    -------
    |dict|_ [|str|_, |list|_ [|int|_]]:
        A dictionary with atomic symbols and a list of matching atomic indices.

    """
    at_list = kf.read('Molecule', 'AtomSymbols').split()
    return group_by_values(enumerate(at_list))
