"""A module for reading multi-xyz files."""

from io import TextIOWrapper
from typing import (Tuple, Dict, Iterable, List)

import numpy as np

__all__ = ['read_multi_xyz']


def read_multi_xyz(filename: str) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Read a (multi) .xyz file.

    Returns the following items:

        * An array with the cartesian coordinates of :math:`m` molecules
          consisting of :math:`n` atoms.

        * A dictionary with atomic symbols and lists of matching atomic indices

    :parameter str filename: The path + filename of a (multi) .xyz file.
    :return: A 3D array with cartesian coordinates and a dictionary
        with atomic symbols as keys and lists of matching atomic indices as values.
    :rtype: :math:`m * n * 3` |np.ndarray|_ [|np.float64|_] and |dict|_
        (keys: |str|_, values: |list|_ [|int|_]).
    """
    # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
    with open(filename, 'r') as f:
        atom_count = _get_atom_count(f)
        idx_dict = _get_idx_dict(f, mol_size=atom_count, subtract=1)
        line_count = _get_line_count(f, add=[2, atom_count])

    # Check if mol_count is fractional, smaller than 1 or if atom_count is smaller than 1
    mol_count = line_count / (2 + atom_count)
    validate_xyz(mol_count, atom_count, filename)

    # Create an empty m*n*3 xyz array
    shape = int(mol_count), atom_count, 3
    xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    with open(filename, 'r') as f:
        for i, _ in enumerate(f):
            next(f)
            xyz[i] = [at.split()[1:] for _, at in zip(range(atom_count), f)]
    return xyz, idx_dict


class XYZError(Exception):
    """Raise when there are issues related to parsing .xyz files."""
    pass


def validate_xyz(mol_count: float,
                 atom_count: int,
                 filename: str) -> None:
    """Validate **mol_count** and **atom_count** in **xyz_file**.

    :parameter float mol_count: The number of molecules in the xyz file.
        Expects float that is finite with integral value (*e.g.* 5.0, 6.0 or 3.0).
    :parameter int atom_count: The number of atoms per molecule.
    :parameter str xyz_file: The path + filename of a (multi) .xyz file.
    """
    if not mol_count.is_integer():
        error = "A non-integer number of molecules ({:d}) was found in '{}'"
        raise XYZError(error.format(mol_count, filename))
    elif mol_count < 1.0:
        error = "No molecules were found in '{}'; mol count: {:f}"
        raise XYZError(error.format(filename, mol_count))
    if atom_count < 1:
        error = "No atoms were found in '{}'; atom count per molecule: {:d}"
        raise XYZError(error.format(filename, atom_count))


def _get_atom_count(f: TextIOWrapper) -> int:
    """Extract the number of atoms per molecule from the first line in an .xyz file.

    :parameter f: An opened .xyz file.
    :type f: |io.TextIOWrapper|_
    :return: The number of atoms per molecule.
    :rtype |int|_
    """
    ret = f.readline()
    try:
        return int(ret)
    except ValueError:
        raise XYZError("{} is not a valid integer, the first line in '{}' should "
                       "contain the number of atoms per molecule".format(ret, f.name))


def _get_line_count(f: TextIOWrapper,
                    add: Iterable[int] = 0) -> int:
    """Extract the total number lines from **f**.

    :parameter f: An opened .xyz file.
    :type f: |io.TextIOWrapper|_
    :parameter add: Add a constant to the to-be returned line count.
    :type add: |int|_ or |list|_ [|int|_]
    :return: The total number of lines in **f**.
    :rtype: |int|_
    """
    for i, _ in enumerate(f, 1):
        pass
    return i + sum(add)


def _get_idx_dict(f: TextIOWrapper,
                  atom_count: int,
                  subtract: int = 0) -> Dict[str, list]:
    """Extract atomic symbols and matching atomic indices from **f**.

    :parameter f: An opened .xyz file.
    :type f: |io.TextIOWrapper|_
    :parameter int atom_count: The number of atoms per molecule in **f**.
    :parameter int subtract: Ignore the first :math:`n` lines in **f**
    :return: A dictionary with atomic symbols and a list of matching atomic indices.
    :rtype: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    """
    idx_dict: Dict[str, List[int]] = {}
    abort = atom_count - subtract
    for i, at in enumerate(f, -subtract):
        if i < 0:  # Skip the header
            continue

        at = at.split()[0].capitalize()
        try:  # Update idx_dict with new atomic indices
            idx_dict[at].append(i)
        except KeyError:
            idx_dict[at] = [i]

        if i == abort:  # If a single molecule has been fully parsed
            break

    for value in idx_dict.values():  # Sort the indices and return
        value.sort()
    return idx_dict
