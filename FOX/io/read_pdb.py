"""
FOX.io.read_pdb
=================

A module for reading protein DataBank (.pdb) files.

Index
-----
.. currentmodule:: FOX.io.read_pdb
.. autosummary::
    read_pdb
    _get_bonds_array
    _get_atoms_df

API
---
.. autofunction:: FOX.io.read_pdb.read_pdb
.. autofunction:: FOX.io.read_pdb._get_bonds_array
.. autofunction:: FOX.io.read_pdb._get_atoms_df

"""


from typing import Tuple, List, Iterable, Sequence

import pandas as pd
import numpy as np

from ..functions.utils import slice_str

__all__ = ['read_pdb']


def read_pdb(filename: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Read the content of protein DataBank file (pdb_).

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html

    Parameters
    ----------
    str filename:
        The path+name of a .pdb file.

    Returns
    -------
    |pd.DataFrame|_ and |np.ndarray|_ [|np.int64|_]:
        A dataframe holding the ATOM/HETATM block and an array holding the CONECT block.

    """
    atoms = []
    bonds = []

    with open(filename, 'r') as f:
        # Intervals for slicing the ATOM/HETATOM block of a .pdb file (see _pdb)
        interval = [None, 6, 11, 16, 17, 20, 22, 26, 27, 38, 46, 54, 60, 66, 73, 78, None]

        # Parse and return
        for i in f:
            if i[0:6] in ('HETATM', 'ATOM  '):
                atoms.append(slice_str(i, interval))
            elif i[0:6] == 'CONECT':
                bonds.append(i.split()[1:])
            elif i[0:4] == 'END':
                break
    return _get_atoms_df(atoms), _get_bonds_array(bonds)


def _get_bonds_array(bonds: Iterable[Sequence[str]]) -> np.ndarray:
    """Convert the connectivity list produced by :func:`.read_pdb` into an array of integers.

    Bond orders are multiplied by :math:`10` and stored as integers,
    thus effectively being stored as floats with single-decimal precision.

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html

    Parameters
    ----------
    bonds : |Iterable|_ [|Sequence|_ [str]]
        A nested list of atomic indices as retrieved from a pdb_ file.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.int64|_]:
        An array with :math:`n` bonds.
        Atomic indices are located in columns 1 & 2 and bond orders, multiplied by 10, are located
        in column 3.

    """
    ret: List[Sequence[object]] = []
    j_old = None
    for i in bonds:
        for j in i[1:]:

            if j is None:  # No bond
                j_old = j
                continue

            elif j == j_old:  # Double bond
                del ret[-1]
                ret.append((i[0], j, 20))

            else:  # Single bond
                ret.append((i[0], j, 10))

            j_old = j

    return np.array(ret, dtype=int)


def _get_atoms_df(atoms: Iterable[Sequence[str]]) -> pd.DataFrame:
    """Convert the atom list produced by :func:`.read_pdb` into a dataframe.

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html

    Parameters
    ----------
    bonds : |Iterable|_ [|Sequence|_ [str]]
        A nested list of atom data as retrieved from a pdb_ file.

    Returns
    -------
    :math:`n*16` |pd.DataFrame|_:
        A dataframe with 16 columns containing the .pdb data of :math:`n` atoms.

    """
    at_columns = {
        'ATOM': object,
        'serial number': int,
        'atom name': object,
        'alternate location': object,
        'residue name': object,
        'chain identifier': object,
        'residue sequence number': int,
        'residue insertions': object,
        'x': float,
        'y': float,
        'z': float,
        'occupancy': float,
        'temperature factor': float,
        'segment identifier': object,
        'Element symbol': object,
        'charge': object
    }

    # Create a dataframe with appropiate columns and data types
    ret = pd.DataFrame(atoms, columns=list(at_columns))
    for key, value in at_columns.items():
        ret[key] = ret[key].astype(value, copy=False)

    # Fix the datatype of the 'charge' column
    charge = ret['charge'].values
    for i, j in enumerate(charge):
        if '-' in j:
            charge[i] = '-' + j.strip('-')
        elif '+' in j:
            charge[i] = j.strip('+')
    charge[charge == ''] = 0.0
    ret['charge'] = charge.astype(float, copy=False)

    return ret
