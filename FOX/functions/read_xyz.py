""" A module for reading multi-xyz files. """

__all__ = ['read_multi_xyz']

import numpy as np


def read_multi_xyz(xyz_file):
    """ Reads a (multi) .xyz file and return a *m*n*3* array with the cartesian coordinates of *m*
    molecules consisting of *n* atoms.

    :parameter str file: The path + filename of a (multi) .xyz file.
    :return: A 3D array of cartesian coordinates and a dictionary object
        with atomic symbols as keys and matching atomic indices as values.
    :rtype: *m*n*3* |np.ndarray|_ [|np.float64|_] and |dict|_
        (keys: |str|_, values: |list|_ [|int|_]).
    """
    # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
    with open(xyz_file, 'r') as file:
        atom_count = _get_mol_size(file)
        idx_dict = _get_idx_dict(file, mol_size=atom_count, subtract=1)
        file_size = _get_file_size(file, add=[2, atom_count])

    # Check if mol_count is fractional, smaller than 1 or if atom_count is smaller than 1
    mol_count = file_size / (2 + atom_count)
    if not mol_count.is_integer():
        error = 'A non-integer number of molecules was found in ' + xyz_file + ': ' + str(mol_count)
        raise IndexError(error)
    elif mol_count < 1.0:
        raise IndexError(str(int(mol_count)) + ' molecules were found in ' + xyz_file)
    if atom_count < 1:
        raise IndexError(str(atom_count) + ' atoms per molecule were found in ' + xyz_file)
    mol_count = int(mol_count)

    # Create an empty (m*n)*3 xyz array
    shape = mol_count, atom_count, 3
    xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    with open(xyz_file, 'r') as f:
        for i, _ in enumerate(f):
            xyz[i] = [at.split()[1:] for _, at in zip(range(atom_count+1), f)][1:]
    return xyz, idx_dict


def _get_mol_size(f):
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
        error = str(ret) + ' is not a valid integer, the first line in an .xyz file should '
        error += 'contain the number of atoms per molecule'
        raise IndexError(error)


def _get_file_size(f, add=0):
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
