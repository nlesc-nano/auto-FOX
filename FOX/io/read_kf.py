"""A module for potential energy surfaces from KF binary files."""

from typing import (Tuple, Dict, List)

import numpy as np

from scm.plams import KFReader

__all__ = ['read_kf']


def read_kf(filename: str) -> Tuple[np.ndarray, Dict[str, List[int]]]:
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
    shape = mol_count, atom_count * 3
    xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    coords = 'Coords({:d})'
    for i in range(mol_count):
        xyz[i] = kf.read('History', coords.format(i+1))
    xyz.shape = mol_count, atom_count, 3

    return xyz, _get_idx_dict(kf)


def _get_idx_dict(kf: KFReader) -> Dict[str, list]:
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

    ret: Dict[str, List[int]] = {}
    for i, at in enumerate(at_list):
        try:
            ret[at].append(i)
        except KeyError:
            ret[at] = [i]
    return ret
