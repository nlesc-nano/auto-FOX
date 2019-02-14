""" A module for reading multi-xyz files. """

__all__ = ['read_multi_xyz']

import numpy as np


def read_multi_xyz(xyz_file, ret_idx_dict=True):
    """ Reads a (multi) .xyz file and return a *m*n*3* array with the cartesian coordinates of *m*
    molecules consisting of *n* atoms.

    :parameter str file: The path + filename of a xyz or multi-xyz file.
    :parameter bool ret_idx_dict: In addition to returning cartesian coordinates, return a
        dictionary with atomic symbols and matching atomic indices (|str|_: |list|_ [|int|_]).
    :return: A *m*n*3* array (|np.ndarray|_) of cartesian coordinates and, optionally, a dictionary
        (|dict|_) with atomic symbols as keys and matching atomic indices as
        values (|str|_: |list|_ [|int|_]).
    """
    with open(xyz_file, 'r') as file:
        # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
        mol_size = get_mol_size(file)
        idx_dict = get_idx_dict(file, mol_size=mol_size, subtract=1)
        file_size = get_file_size(file, add=[2, mol_size])
        mol_count_float = file_size / (2 + mol_size)
        mol_count = int(mol_count_float)

        # Check if mol_count_float is fractional; raise an error if it is
        if mol_count_float - mol_count != 0.0:
            error = 'A non-integer number of molecules was found in '
            error += xyz_file + ': ' + str(mol_count)
            raise IndexError(error)

        # Create an empty (m*n)*3 xyz array
        shape = mol_count * mol_size, 3
        xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    with open(xyz_file, 'r') as file:
        j = 0
        for i, at in enumerate(file):
            at = at.split()
            if len(at) == 4:
                xyz[i-j] = at[1:]
            else:
                j += 1

    # Return the xyz array or the xyz array and a dictionary: {atomic symbols: [atomic indices]}
    xyz.shape = mol_count, mol_size, 3
    if ret_idx_dict:
        return xyz, idx_dict
    return xyz


def get_mol_size(file):
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


def get_file_size(file, add=0):
    """ Extract the total number lines from a text file.
    file <_io.TextIOWrapper>: An opened text file.
    add <int>: An <int> or iterable consisting of <int>; adds a constant to the number of lines.
    return <int>: The number of lines in a text file.
    """
    for i, _ in enumerate(file, 1):
        pass
    return i + sum(add)


def get_idx_dict(file, mol_size=False, subtract=0):
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
                return idx_dict
