""" A module for reading multi-xyz files. """

__all__ = ['read_multi_xyz', 'grab_random_slice', 'multi_xyz_to_molecule']

import numpy as np

from scm.plams import (Molecule, Atom)


def read_multi_xyz(xyz_file, ret_idx_dict=True):
    """ Reads a (multi) .xyz file and return a *m*n*3* array with the cartesian coordinates of *m*
    molecules consisting of *n* atoms.

    :parameter str file: The path + filename of a (multi) .xyz file.
    :parameter bool ret_idx_dict: In addition to returning cartesian coordinates, return a
        dictionary with atomic symbols and matching atomic indices (|str|_: |list|_ [|int|_]).
    :return: A 3D array of cartesian coordinates and, optionally, a dictionary
        with atomic symbols as keys and matching atomic indices as alues.
    :rtype: *m*n*3* |np.ndarray|_ [|np.float64|_] and, optionally, |dict|_
        (keys: |str|_, values: |list|_ [|int|_]).
    """
    # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
    with open(xyz_file, 'r') as file:
        mol_size = get_mol_size(file)
        idx_dict = get_idx_dict(file, mol_size=mol_size, subtract=1)
        file_size = get_file_size(file, add=[2, mol_size])

    # Check if mol_count_float is fractional; raise an error if it is
    mol_count = file_size / (2 + mol_size)
    if not mol_count.is_integer():
        error = 'A non-integer number of molecules was found in '
        error += xyz_file + ': ' + str(mol_count)
        raise IndexError(error)
    mol_count = int(mol_count)

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


def grab_random_slice(xyz_array, p=0.5):
    """ Grab and return a random number of 2D slices from **xyz_array** as a new array.
    Slices are grabbed along axis 0. The new array consists of views of **xyz_array**.

    :parameter xyz_array: A 3D array with cartesian coordinates of *m*
        molecules consisting of *n* atoms.
    :type xyz_array: *m*n*3* |np.ndarray|_ [|np.float64|_]
    :parameter float p: The probability of returning each 2D slice from **xyz_array**.
        Accepts values between 0.0 (0%) and 1.0 (100%).
    :return: A random amount of 2D slices, weighted by **p**, from **xyz_array**.
    :rtype: *k*n*3* |np.ndarray|_ [|np.float64|_], where *k* ≈ *m* * **p**.
    """
    if p <= 0.0 or p >= 1.0:
        raise IndexError('The probability, p, must be larger than 0.0 and smaller than 1.0')
    elif len(xyz_array.shape) == 2 or xyz_array.shape[0] == 1:
        raise IndexError('Grabbing random 2D slices from a 2D array makes no sense')

    size = int(xyz_array.shape[0] * p)
    idx_range = np.arange(xyz_array.shape[0], dtype=int)
    idx = np.random.choice(idx_range, size)
    return xyz_array[idx]


def multi_xyz_to_molecule(xyz_array, idx_dict):
    """ Convert the output of :func:`FOX.functions.read_xyz.read_multi_xyz`, an array and
    dictionary, into a list of PLAMS molecules.

    :parameter xyz_array: A 3D or 2D array with cartesian coordinates of *m*
        molecules consisting of *n* atoms.
    :type xyz_array: *m*n*3* or *n*3* |np.ndarray|_ [|np.float64|_]
    :parameter idx_dict: A dictionary with the atomic symbols in **xyz_array** as keys
        and matching atomic indices as values.
    :type idx_dict: |dict|_ (keys: |str|_, values: |list|_ [|int|_]).
    :return: A list of PLAMS molecules.
    :rtype: |list|_ [|plams.Molecule|_].
    """
    # Make sure we're dealing with a 3d array
    if len(xyz_array.shape) == 2:
        xyz_array = xyz_array[None, :, :]

    # Create a dictionary with atomic indices as keys and matching atomic symbols as values
    idx_dict = invert_dict(idx_dict)

    # Construct a template molecule
    mol_template = Molecule()
    mol_template.properties.frame = 1
    for i in range(xyz_array.shape[1]):
        at = Atom(symbol=idx_dict[i])
        mol_template.add_atom(at)

    # Create copies of the template molecule and update their cartesian coordinates
    ret = []
    for i, xyz in enumerate(xyz_array):
        mol = mol_template.copy()
        mol.from_array(xyz)
        mol.properties.frame += i
        ret.append(mol)

    return ret


def invert_dict(dic):
    """ Take a dictionary and turn keys into values and values into keys. """
    ret = {}
    for key in dic:
        for i in dic[key]:
            ret[i] = key
    return ret


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
