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
    elif mol_count < 1:
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


def _get_mol_size(file):
    """ Extract the number of atoms in a molecule from an .xyz file.

    The number of atoms is extracted form the first line.
    file <_io.TextIOWrapper>: An opened text file.
    return <int>: The number of atoms per molecule.
    """
    item = file.readline()
    try:
        return int(item)
    except ValueError:
        error = str(item) + ' is not a valid integer, the first line in an .xyz file should '
        error += 'contain the number of atoms in a molecule'
        raise IndexError(error)


def _get_file_size(file, add=0):
    """ Extract the total number lines from a text file.

    file <_io.TextIOWrapper>: An opened text file.
    add <int>: An <int> or iterable consisting of <int>; adds a constant to the number of lines.
    return <int>: The number of lines in a text file.
    """
    for i, _ in enumerate(file, 1):
        pass
    return i + sum(add)


def _get_idx_dict(file, mol_size=False, subtract=0):
    """ Extract atomic symbols from an opened text file.

    file <_io.TextIOWrapper>: An opened text file.
    mol_size <int>: The number of atoms in a single molecule.
    subtract <int>: Ignore the first n lines in *file*
    return <dict>: A dictionary {atomic symbols: [atomic indices]}.
    """
    idx_dict = {}
    abort = mol_size - subtract
    for i, at in enumerate(file, -subtract):
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
