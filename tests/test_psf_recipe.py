"""Tests for :mod:`FOX.recipes.psf`."""

from os.path import join

import numpy as np

from scm.plams import Molecule, MoleculeError
from assertionlib import assertion

from FOX.recipes import generate_psf, extract_ligand

PATH: str = join('tests', 'test_files')
STR_FILE: str = join(PATH, 'ligand.str')
QD: Molecule = Molecule(join(PATH, 'Cd68Se55_26COO_MD_trajec.xyz'))


def test_extract_ligand() -> None:
    """Tests for :func:`extract_ligand`."""
    ligand = extract_ligand(QD, 4, {'C', 'H', 'O'})
    ref = np.array([[5.47492504, -3.54070096, -3.91403079],
                    [6.25736402, -3.70922481, -4.71302813],
                    [4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    ligand = extract_ligand(QD, 2, 'O')
    ref = np.array([[4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    assertion.assert_(extract_ligand, QD, 4, 'bob', exception=MoleculeError)


def test_generate_psf() -> None:
    """Tests for :func:`extract_ligand`."""
    ligand = extract_ligand(QD, 4, {'C', 'H', 'O'})
    psf = generate_psf(QD, ligand, str_file=STR_FILE)
    ref = np.load(join(PATH, 'generate_psf_bonds.npy'))

    np.testing.assert_array_equal(psf.bonds, ref)
