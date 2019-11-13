"""A module for testing files in the :mod:`FOX.io.read_pdb` module."""

from os.path import join

import pandas as pd
import numpy as np

from FOX.io.read_pdb import read_pdb

PATH = join('tests', 'test_files')


def test_read_pdb():
    """Test :func:`FOX.io.read_pdb.read_pdb`."""
    atoms, bonds = read_pdb(join(PATH, 'mol.pdb'))
    ref_bonds = np.load(join(PATH, 'pdb_bonds.npy'))
    ref_atoms = pd.read_csv(
        join(PATH, 'pdb_atoms.csv'), float_precision='high', index_col=0, keep_default_na=False
    )

    for key in atoms:
        i, j = atoms[key], ref_atoms[key]
        np.testing.assert_array_equal(i, j)
    np.testing.assert_array_equal(atoms.index, ref_atoms.index)
    np.testing.assert_array_equal(atoms.columns, ref_atoms.columns)
    np.testing.assert_array_equal(bonds, ref_bonds)
