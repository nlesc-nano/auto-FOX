""" A module for reading protein DataBank (.pdb) files.
 """

__all__ = []

import pandas as pd
import numpy as np


def slice_str(str_, intervals):
    """ Slice a string, **str_**, at intervals specified in **intervals**.

    :parameter str str_: A string.
    :parameter list intverals: A list with :math:`n` objects suitable for slicing.
    :return: A list of strings as sliced from **str_**.
    :rtype: :math:`n-1` |list|_ [|str|_]
    """
    iter1 = intervals[:-1]
    iter2 = intervals[1:]
    return [str_[i:j].strip() for i, j in zip(iter1, iter2)]


def read_pdb(pdb_file):
    """ Read the content of protein DataBank file (pdb_).

    :parameter str pdb_file: The path+name of a .pdb file.
    :return: A dataframe holding the ATOM/HETATM block and an array holding the CONECT block.
    :rtype: |pd.DataFrame|_ and |np.ndarray|_ [|np.int64|_]

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """
    atoms = []
    bonds = []

    with open(pdb_file, 'r') as f:
        # Intervals for slicing the ATOM/HETATOM block of a .pdb file
        interval = [None, 6, 11, 16, 17, 20, 22, 26, 27, 38, 46, 54, 60, 66, 73, 78, None]

        # Parse and return
        for i in f:
            if i[0:6] == 'HETATM' or i[0:4] == 'ATOM':
                atoms.append(slice_str(i, interval))
            elif i[0:6] == 'CONECT':
                bonds.append(i.split()[1:])
            elif i[0:4] == 'END':
                break
    return _get_atoms_df(atoms), _get_bonds_array(bonds)


def _get_bonds_array(bonds):
    """ Convert the connectivity list produced by :func:`read_pdb` into an array of integers.

    :parameter list bonds: A nested list of atomic indices as retrieved from a pdb_ file.
    :return: An array with :math:`n` bonds.
        Atomic indices are located in columns 1 & 2 and bond orders, multiplied by 10, are located
        in column 3.
    :rtype: :math:`n*3` |np.ndarray|_ [|np.int64|_]

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """
    ret = []
    j_old = None
    for i in bonds:
        for j in i[1:]:
            if j is None:
                j_old = j
                continue
            if j == j_old:
                del ret[-1]
                ret.append((i[0], j, 20))
            else:
                ret.append((i[0], j, 10))
            j_old = j

    return np.array(ret, int)


def _get_atoms_df(atoms):
    """ Convert the atom list produced by :func:`read_pdb` into a dataframe.

    :parameter list atoms: A nested list of atom data as retrieved from a pdb_ file.
    :return: A dataframe with 16 columns containing the .pdb data of :math:`n` atoms.
    :rtype: :math:`n*16` |pd.DataFrame|_

    .. _pdb: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
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
