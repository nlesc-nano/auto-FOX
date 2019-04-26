""" A module for reading multi-xyz files. """

__all__ = ['read_multi_xyz']

import numpy as np


def read_multi_xyz(xyz_file):
    """ Read a (multi) .xyz file and return:

        * An array with the cartesian coordinates of *m* molecules consisting of *n* atoms.

        * A dictionary with atomic symbols and lists of matching atomic indices

    :parameter str xyz_file: The path + filename of a (multi) .xyz file.
    :return: A 3D array with cartesian coordinates and a dictionary
        with atomic symbols as keys and lists of matching atomic indices as values.
    :rtype: *m*n*3* |np.ndarray|_ [|np.float64|_] and |dict|_
        (keys: |str|_, values: |list|_ [|int|_]).
    """
    # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
    with open(xyz_file, 'r') as f:
        atom_count = _get_atom_count(f)
        idx_dict = _get_idx_dict(f, mol_size=atom_count, subtract=1)
        line_count = _get_line_count(f, add=[2, atom_count])

    # Check if mol_count is fractional, smaller than 1 or if atom_count is smaller than 1
    mol_count = line_count / (2 + atom_count)
    validate_xyz(mol_count, atom_count, xyz_file)

    # Create an empty (m*n)*3 xyz array
    shape = int(mol_count), atom_count, 3
    xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    with open(xyz_file, 'r') as f:
        for i, _ in enumerate(f):
            next(f)
            xyz[i] = [at.split()[1:] for _, at in zip(range(atom_count), f)]
    return xyz, idx_dict


class XYZError(Exception):
    """ Raise when there are issues related to parsing .xyz files. """
    pass


def validate_xyz(mol_count, atom_count, xyz_file):
    """ Validate **mol_count** and **atom_count** in **xyz_file**.

    :parameter float mol_count: The number of molecules in the xyz file.
        Expects float that is finite with integral value (*e.g.* 5.0, 6.0 or 3.0).
    :parameter int atom_count: The number of atoms per molecule.
    :parameter str xyz_file: The path + filename of a (multi) .xyz file.
    """
    if not mol_count.is_integer():
        error = "A non-integer number of molecules ({:d}) was found in '{}'"
        raise XYZError(error.format(mol_count, xyz_file))
    elif mol_count < 1.0:
        error = "No molecules were found in '{}'; mol count: {:f}"
        raise XYZError(error.format(xyz_file, mol_count))
    if atom_count < 1:
        error = "No atoms were found in '{}'; atom count per molecule: {:d}"
        raise XYZError(error.format(xyz_file, atom_count))


def _get_atom_count(f):
    """ Extract the number of atoms per molecule from the first line in an .xyz file.

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


def _get_line_count(f, add=0):
    """ Extract the total number lines from **f**.

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


def _get_idx_dict(f, mol_size, subtract=0):
    """ Extract atomic symbols and matching atomic indices from **f**.

    :parameter f: An opened .xyz file.
    :type f: |io.TextIOWrapper|_
    :parameter int mol_size: The number of atoms per molecule in **f**.
    :subtract: Ignore the first n lines in **f**
    :return: A dictionary with atomic symbols and a list of matching atomic indices.
    :rtype: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    """
    idx_dict = {}
    abort = mol_size - subtract
    for i, at in enumerate(f, -subtract):
        if i >= 0:
            at = at.split()[0].capitalize()
            try:
                idx_dict[at].append(i)
            except KeyError:
                idx_dict[at] = [i]
            if i == abort:
                for key in idx_dict:
                    idx_dict[key] = sorted(idx_dict[key])
                return idx_dict
