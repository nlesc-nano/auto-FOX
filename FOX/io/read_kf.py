""" A module for reading ADF binary files. """

__all__ = ['read_kf']

import numpy as np

from scm.plams import KFFile


def read_kf(filename):
    """ Read an ADF binary file.
    """
    kf = KFFile(filename)
    atom_count = kf.read('Molecule', 'nAtoms')
    mol_count = kf.read('History', 'nEntries')

    shape = mol_count, atom_count * 3
    xyz = np.empty(shape)

    coords = 'Coords({:d})'
    for i in range(mol_count):
        xyz[i] = kf.read('History', coords.format(i+1))

    xyz.shape = mol_count, atom_count, 3
    return xyz


path = '/Users/basvanbeek/Documents/ADF_DATA/md_test.results/ams.rkf'
xyz = read_kf(path)
