"""A module for testing files in the :class:`FOX.classes.psf.PSF` class."""

from os import remove
from os.path import join

import numpy as np
import pandas as pd

from FOX.classes.psf import PSF

__all__: list = []

REF_DIR = 'tests/test_files/psf'


def test_read_psf():
    """ Test :meth:`FOX.classes.psf.PSF.read_psf`. """
    psf_dict = PSF.read_psf(join(REF_DIR, 'mol.psf'))

    ref_atoms = pd.read_csv(join(REF_DIR, 'atoms.csv'), float_precision='high', index_col=0)
    ref_dict = {
        'bonds': np.load(join(REF_DIR, 'bonds.npy')),
        'angles': np.load(join(REF_DIR, 'angles.npy')),
        'dihedrals': np.load(join(REF_DIR, 'dihedrals.npy')),
        'impropers': np.load(join(REF_DIR, 'impropers.npy')),
        'donors': np.load(join(REF_DIR, 'donors.npy')),
        'acceptors': np.load(join(REF_DIR, 'acceptors.npy')),
        'no_nonbonded': np.load(join(REF_DIR, 'no_nonbonded.npy')),
    }

    assert (
        psf_dict.title ==
        np.array(['PSF file generated with Auto-FOX:', 'https://github.com/nlesc-nano/auto-FOX'])
    ).all()

    for key, value in psf_dict.atoms.items():
        i, j = value, ref_atoms[key]
        try:
            np.testing.assert_allclose(i, j)
        except TypeError:
            np.testing.assert_array_equal(i, j)

    np.testing.assert_array_equal(psf_dict.atoms.index, ref_atoms.index)
    np.testing.assert_array_equal(psf_dict.atoms.columns, ref_atoms.columns)

    for key, value in ref_dict.items():
        i, j = value, getattr(psf_dict, key)
        np.testing.assert_array_equal(i, j)


def test_write_psf():
    """ Test :meth:`FOX.classes.psf.PSF.write_psf`. """
    psf_dict = PSF.read_psf(join(REF_DIR, 'mol.psf'))
    psf_dict.write_psf(join(REF_DIR, 'mol_test.psf'))
    psf_dict = PSF.read_psf(join(REF_DIR, 'mol_test.psf'))
    remove(join(REF_DIR, 'mol_test.psf'))

    ref_atoms = pd.read_csv(join(REF_DIR, 'atoms.csv'), float_precision='high', index_col=0)
    ref_dict = {
        'bonds': np.load(join(REF_DIR, 'bonds.npy')),
        'angles': np.load(join(REF_DIR, 'angles.npy')),
        'dihedrals': np.load(join(REF_DIR, 'dihedrals.npy')),
        'impropers': np.load(join(REF_DIR, 'impropers.npy')),
        'donors': np.load(join(REF_DIR, 'donors.npy')),
        'acceptors': np.load(join(REF_DIR, 'acceptors.npy')),
        'no_nonbonded': np.load(join(REF_DIR, 'no_nonbonded.npy')),
    }

    assert (
        psf_dict.title ==
        np.array(['PSF file generated with Auto-FOX:', 'https://github.com/nlesc-nano/auto-FOX'])
    ).all()

    for key, value in psf_dict.atoms.items():
        i, j = value, ref_atoms[key]
        try:
            np.testing.assert_allclose(i, j)
        except TypeError:
            np.testing.assert_array_equal(i, j)

    np.testing.assert_array_equal(psf_dict.atoms.index, ref_atoms.index)
    np.testing.assert_array_equal(psf_dict.atoms.columns, ref_atoms.columns)

    for key, value in ref_dict.items():
        i, j = value, getattr(psf_dict, key)
        np.testing.assert_array_equal(i, j)
