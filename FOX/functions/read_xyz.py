""" A module for reading multi-xyz files. """

__all__ = ['read_multi_xyz']

import numpy as np


def read_multi_xyz(xyz_file, ret_idx_dict=True):
    """ Return a m*n*3 array of cartesian coordinates extracted from an xyz or multi-xyz file.

    file <str>: The path + filename of a xyz or multi-xyz file.
    ret_idx_dict <bool>: Return a dictionary consisting of {atomic symbols: [atomic indices]} as
        derived from the first molecule in *file*.
    return <np.ndarray> (and <dict>): A m*n*3 numpy array with the cartesian coordinates of
        m molecules (isomers) consisting of n atoms. Optionally, a dictionary can be returned
        consisting of {atomic symbols: [atomic indices]}.
    """
    with open(xyz_file, 'r') as file:
        # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
        mol_size = get_mol_size(file)
        idx_dict = get_idx_dict(file, mol_size)
        file_size = get_file_size(file, add=[2, mol_size])
        mol_count = file_size // (2 + mol_size)

        # Create an empty (m*n)*3 xyz array
        shape = mol_count * mol_size, 3
        xyz = np.empty(shape)

    # Fill the xyz array with cartesian coordinates
    with open(xyz_file, 'r') as file:
        j=0
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
    return <int>: The number of atoms per mol.
    """
    for item in file:
        try:
            return int(item)
        except ValueError:
            pass


def get_file_size(file, add=0):
    """ Extract the total number lines from a text file.
    file <_io.TextIOWrapper>: An opened text file.
    add <int>: An <int> or iterable consisting of <int>; adds a constant to the number of lines.
    return <int>: The number of lines in a text file.
    """
    for i, _ in enumerate(file, 1):
        pass
    return i + sum(add)


def get_idx_dict(file, mol_size=False):
    """ Extract atomic symbols from an opened text file.
    file <_io.TextIOWrapper>: An opened text file.
    mol_size <int>
    return <dict>: A dictionary {atomic symbols: [atomic indices]}
    """
    idx_dict = {}
    abort = mol_size - 1
    for i, at in enumerate(file, -1):
        if i >= 0:
            at = at.split()[0].capitalize()
            try:
                idx_dict[at].append(i)
            except KeyError:
                idx_dict[at] = [i]
            if i == abort:
                return idx_dict
