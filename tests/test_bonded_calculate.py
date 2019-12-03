"""A module for testing :mod:`FOX.ff.bonded_calculate`."""

from pathlib import Path

import numpy as np
from assertionlib import assertion

from FOX import MultiMolecule, get_bonded

PATH = Path('tests') / 'test_files'


def test_get_bonded() -> None:
    """Test :func:`FOX.functions.lj_calculate.get_non_bonded`."""
    psf = PATH / 'Cd68Se55_26COO_MD_trajec.psf'
    prm = PATH / 'Cd68Se55_26COO_MD_trajec.prm'
    mol = MultiMolecule.from_xyz(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')

    ref1 = np.load(PATH / 'get_bonded.bonds.npy')
    ref2 = np.load(PATH / 'get_bonded.angles.npy')
    ref4 = np.load(PATH / 'get_bonded.impropers.npy')
    angles, bonds, dihedrals, impropers = get_bonded(mol, psf, prm)

    np.testing.assert_allclose(bonds, ref1)
    np.testing.assert_allclose(angles, ref2)
    assertion.is_(dihedrals, None)
    np.testing.assert_allclose(impropers, ref4)
