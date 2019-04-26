""" A module for testing the FOX.MultiMolecule class. """

__all__ = []

from os import remove
from os.path import join

import pytest
import numpy as np

from FOX import (MultiMolecule, get_example_xyz)


MOL = MultiMolecule.from_xyz(get_example_xyz())
REF_DIR = 'test/test_files'


def test_guess_bonds():
    """ Test :meth:`FOX.MultiMolecule.guess_bonds`. """
    mol = MOL.copy()

    mol.guess_bonds(atom_subset=['H', 'C', 'O'])
    ref = np.load(join(REF_DIR, 'guess_bonds.npy'))
    np.testing.assert_allclose(mol.bonds, ref)


def test_slice():
    """ Test :meth:`FOX.MultiMolecule.slice`. """
    mol = MOL.copy()

    mol.slice(start=0, stop=None, step=1, inplace=True)
    np.testing.assert_allclose(mol, MOL)

    mol_new = mol.slice(start=1000, stop=None, step=1)
    np.testing.assert_allclose(mol_new, mol[1000:])

    mol_new = mol.slice(start=0, stop=1000, step=1)
    np.testing.assert_allclose(mol_new, mol[0:1000])

    mol_new = mol.slice(start=0, stop=None, step=10)
    np.testing.assert_allclose(mol_new, mol[0::10])


def test_random_slice():
    """ Test :meth:`FOX.MultiMolecule.random_slice`. """
    mol = MOL.copy()

    try:
        mol.random_slice(start=0, stop=None, p=1.0, inplace=True)
    except ValueError:
        pass

    mol_new = mol.random_slice(start=0, stop=None, p=0.5)
    assert mol_new.shape[1:2] == mol.shape[1:2]
    assert mol_new.shape[0] == mol.shape[0] // 2


def test_reset_origin():
    """ Test :meth:`FOX.MultiMolecule.reset_origin`. """
    mol = MOL.copy()

    mol.reset_origin()
    assert mol.mean(axis=1).sum() <= 10**-8

    mol_new = mol.reset_origin(mol_subset=1000, inplace=False)
    assert mol_new[0:1000].mean(axis=1).sum() <= 10**-8

    mol_new = mol.reset_origin(atom_subset=slice(0, 100), inplace=False)
    assert mol_new[:, 0:100].mean(axis=1).sum() <= 10**-8


def test_sort():
    """ Test :meth:`FOX.MultiMolecule.sort`. """
    mol = MOL.copy()

    mol.sort(sort_by='symbol')
    np.testing.assert_array_equal(mol.symbol, np.sort(mol.symbol))

    mol.sort(sort_by='atnum')
    np.testing.assert_allclose(mol.atnum, np.sort(mol.atnum))

    mol.sort(sort_by='mass')
    np.testing.assert_allclose(mol.mass, np.sort(mol.mass))

    mol.sort(sort_by='radius')
    np.testing.assert_allclose(mol.radius, np.sort(mol.radius))

    mol.sort(sort_by='connectors')
    np.testing.assert_allclose(mol.connectors, np.sort(mol.connectors))


def test_residue_argsort():
    """ Test :meth:`FOX.MultiMolecule.residue_argsort`. """
    mol = MOL.copy()

    mol.guess_bonds(atom_subset=['H', 'C', 'O'])
    idx = mol.residue_argsort()
    ref = np.load(join(REF_DIR, 'residue_argsort.npy'))
    np.testing.assert_allclose(idx, ref)


def test_get_center_of_mass():
    """ Test :meth:`FOX.MultiMolecule.sort`. """
    mol = MOL.copy()

    center_of_mass = mol.get_center_of_mass()
    mol -= center_of_mass[:, None, :]
    center_of_mass = mol.get_center_of_mass()
    assert np.abs(center_of_mass.mean()) <= 10**-8


def test_get_bonds_per_atom():
    """ Test :meth:`FOX.MultiMolecule.get_bonds_per_atom`. """
    mol = MOL.copy()

    mol.guess_bonds(atom_subset=['H', 'C', 'O'])
    bond_count = mol.get_bonds_per_atom()
    ref = np.load(join(REF_DIR, 'get_bonds_per_atom.npy'))
    np.testing.assert_allclose(bond_count, ref)


def test_rdf():
    """ Test :meth:`FOX.MultiMolecule.init_rdf`. """
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rdf = mol.init_rdf(atom_subset=atoms).values
    ref = np.load(join(REF_DIR, 'rdf.npy'))
    np.testing.assert_allclose(rdf, ref)


def test_rmsf():
    """ Test :meth:`FOX.MultiMolecule.init_rmsf`. """
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rmsf = mol.init_rmsf(atom_subset=atoms).values
    np.nan_to_num(rmsf, copy=False)
    ref = np.load(join(REF_DIR, 'rmsf.npy'))
    np.nan_to_num(ref, copy=False)
    np.testing.assert_allclose(rmsf, ref)


def test_rmsd():
    """ Test :meth:`FOX.MultiMolecule.init_rmsd`. """
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rmsd = mol.init_rmsd(atom_subset=atoms).values
    ref = np.load(join(REF_DIR, 'rmsd.npy'))
    np.testing.assert_allclose(rmsd, ref)


@pytest.mark.slow
def test_adf():
    """ Test :meth:`FOX.MultiMolecule.init_adf`. """
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    adf = mol.init_adf(atom_subset=atoms).values
    ref = np.load(join(REF_DIR, 'adf.npy'))
    np.testing.assert_allclose(adf, ref)


def test_as_psf():
    """ Test :meth:`FOX.MultiMolecule.as_psf`. """
    mol = MOL.copy()

    mol.guess_bonds(atom_subset=['H', 'C', 'O'])
    mol.as_psf(join(REF_DIR, 'mol.psf'))
    with open(join(REF_DIR, 'mol.psf')) as f:
        psf = f.read()
    remove(join(REF_DIR, 'mol.psf'))
    with open(join(REF_DIR, 'as_psf.psf')) as f:
        ref = f.read()
    assert psf == ref


def test_as_mass_weighted():
    """ Test :meth:`FOX.MultiMolecule.as_mass_weighted`. """
    mol = MOL.copy()

    mol_new = mol.as_mass_weighted()
    mol *= mol.mass[None, :, None]
    np.testing.assert_allclose(mol, mol_new)


def test_from_mass_weighted():
    """ Test :meth:`FOX.MultiMolecule.from_mass_weighted`. """
    mol = MOL.copy()

    mol_new = mol.as_mass_weighted()
    mol_new.from_mass_weighted()
    assert np.abs((mol_new - mol).mean()) < 10**-8


def test_as_Molecule():
    """ Test :meth:`FOX.MultiMolecule.as_Molecule`. """
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_array = np.array([i.as_array() for i in mol_list])
    np.testing.assert_allclose(mol_array, mol)


def test_from_Molecule():
    """ Test :meth:`FOX.MultiMolecule.from_Molecule`. """
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_new = MultiMolecule.from_Molecule(mol_list)
    np.testing.assert_allclose(mol_new, mol)


def test_as_xyz():
    """ Test :meth:`FOX.MultiMolecule.as_xyz`. """
    mol = MOL.copy()

    xyz = join(REF_DIR, 'mol.xyz')
    mol.as_xyz(filename=xyz)
    mol_new = MultiMolecule.from_xyz(xyz)
    remove(xyz)
    np.testing.assert_allclose(mol_new, mol)


def test_from_xyz():
    """ Test :meth:`FOX.MultiMolecule.from_xyz`. """
    mol = MOL.copy()

    mol_new = MultiMolecule.from_xyz(get_example_xyz())
    np.testing.assert_allclose(mol_new, mol)
