"""A module for testing the :class:`FOX.Molecule` class."""

from os.path import join

import numpy as np

from FOX import (MultiMolecule, get_example_xyz)

__all__: list = []

MOL = MultiMolecule.from_xyz(get_example_xyz())
MOL.guess_bonds(atom_subset=['C', 'O', 'H'])
PLAMS_MOL = MOL.as_Molecule(0)[0]
REF_DIR = 'tests/test_files'


def test_set_atoms_id():
    """Test :meth:`.Molecule.set_atoms_id`."""
    mol = PLAMS_MOL.copy()

    mol.set_atoms_id()
    for i, at in enumerate(mol.atoms, 1):
        assert at.id == i

    mol.set_atoms_id(start=10)
    for i, at in enumerate(mol.atoms, 10):
        assert at.id == i


def test_separate_mod():
    """Test :meth:`.Molecule.separate_mod`."""
    mol = PLAMS_MOL.copy()

    idx = mol.separate_mod()
    j = np.array([len(i) for i in idx])
    ref = np.load(join(REF_DIR, 'separate_mod.npy'))
    np.testing.assert_allclose(j, ref)


def test_fix_bond_orders():
    """Test :meth:`.Molecule.fix_bond_orders`."""
    mol = PLAMS_MOL.copy()

    mol.fix_bond_orders()
    charge = np.array([at.properties.charge for at in mol])
    ref = np.load(join(REF_DIR, 'fix_bond_orders.npy'))
    np.testing.assert_allclose(charge, ref)


def test_get_angles():
    """Test :meth:`.Molecule.get_angles`."""
    mol = PLAMS_MOL.copy()

    angles = mol.get_angles()
    ref = np.load(join(REF_DIR, 'angles.npy'))
    np.testing.assert_allclose(angles, ref)


def test_get_dihedrals():
    """Test :meth:`.Molecule.get_dihedrals`."""
    mol = PLAMS_MOL.copy()

    dihedrals = mol.get_dihedrals()
    ref = np.load(join(REF_DIR, 'dihedrals.npy'))
    np.testing.assert_allclose(dihedrals, ref)


def test_get_impropers():
    """Test :meth:`.Molecule.get_impropers`."""
    mol = PLAMS_MOL.copy()

    impropers = mol.get_impropers()
    ref = np.load(join(REF_DIR, 'impropers.npy'))
    np.testing.assert_allclose(impropers, ref)
