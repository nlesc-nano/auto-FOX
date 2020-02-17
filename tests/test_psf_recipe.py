"""Tests for :mod:`FOX.recipes.psf`."""

import warnings
from os.path import join

import numpy as np

from scm.plams import Molecule, MoleculeError
from assertionlib import assertion

from FOX.recipes import generate_psf, generate_psf2, extract_ligand

PATH: str = join('tests', 'test_files')
STR_FILE: str = join(PATH, 'ligand.str')
QD: Molecule = Molecule(join(PATH, 'Cd68Se55_26COO_MD_trajec.xyz'))


def test_extract_ligand() -> None:
    """Tests for :func:`extract_ligand`."""
    qd = QD.copy()
    ligand = extract_ligand(qd, 4, {'C', 'H', 'O'})
    ref = np.array([[5.47492504, -3.54070096, -3.91403079],
                    [6.25736402, -3.70922481, -4.71302813],
                    [4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    ligand = extract_ligand(qd, 2, 'O')
    ref = np.array([[4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    assertion.assert_(extract_ligand, qd, 4, 'bob', exception=MoleculeError)


def test_generate_psf() -> None:
    """Tests for :func:`extract_ligand`."""
    qd = QD.copy()
    ligand = extract_ligand(qd, 4, {'C', 'H', 'O'})
    psf = generate_psf(qd, ligand, str_file=STR_FILE)
    ref = np.load(join(PATH, 'generate_psf_bonds.npy'))

    np.testing.assert_array_equal(psf.bonds, ref)


def test_generate_psf2() -> None:
    """Tests for :func:`extract_ligand`."""
    qd = QD.copy()
    psf = generate_psf2(qd, 'C(=O)[O-]')
    ref = np.load(join(PATH, 'generate_psf_bonds2.npy'))

    np.testing.assert_array_equal(psf.bonds, ref)

    qd = QD.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_list = generate_psf2(qd, 'CC(=O)[O-]', ret_failed_lig=True)

    assertion.len_eq(mol_list, 1)
    assertion.eq(mol_list[0].get_formula(), 'C2H2O3')
