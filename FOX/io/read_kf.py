""" A module for potential energy surfaces from KF binary files. """

__all__ = ['read_kf']

import numpy as np

from scm.plams import KFReader


def read_kf(filename):
    """ Read a KF binary file, containing a potential energy surface, and return:

        * An array with the cartesian coordinates of :math:`m` molecules
          consisting of :math:`n` atoms.

        * A dictionary with atomic symbols and lists of matching atomic indices.

    Current KF file support is limited to those produced with the AMS_ engine.

    :parameter str filename: The path + filename of the kf file.
    :return: A 3D array with cartesian coordinates and a dictionary
        with atomic symbols as keys and lists of matching atomic indices as values.
    :rtype: :math:`m*n*3` |np.ndarray|_ [|np.float64|_] and |dict|_
        (keys: |str|_, values: |list|_ [|int|_]).

    .. _ams: https://www.scm.com/product/ams/
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


def _get_idx_dict(kf):
    """ Extract atomic symbols and matching atomic indices from **kf**.

    :parameter kf: A KFReader instance constructed from a KF binary file.
    :type kf: |plams.KFReader|_
    :return: A dictionary with atomic symbols and a list of matching atomic indices.
    :rtype: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    """
    at_list = kf.read('Molecule', 'AtomSymbols').split()

    ret = {}
    for i, at in enumerate(at_list):
        try:
            ret[at].append(i)
        except KeyError:
            ret[at] = [i]
    return ret
