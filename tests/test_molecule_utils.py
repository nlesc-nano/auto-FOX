"""A module for testing :mod:`FOX.functions.molecule_utils`."""

from os.path import join

import numpy as np

from scm.plams import Molecule

from FOX.functions.molecule_utils import get_bonds, get_angles, get_dihedrals, get_impropers

PATH: str = join('tests', 'test_files')
MOL: Molecule = Molecule(join(PATH, 'hexanoic_acid.pdb'))
MOL.guess_bonds()


def test_get_bonds() -> None:
    """Tests for :func:`nanoCAT.ff.mol_topology.get_bonds`."""
    mol = MOL.copy()
    bonds_ref = np.load(join(PATH, 'get_bonds.npy'))
    bonds = get_bonds(mol)
    np.testing.assert_array_equal(bonds, bonds_ref)


def test_get_angles() -> None:
    """Tests for :func:`nanoCAT.ff.mol_topology.get_angles`."""
    mol = MOL.copy()
    angles_ref = np.load(join(PATH, 'get_angles.npy'))
    angles = get_angles(mol)
    np.testing.assert_array_equal(angles, angles_ref)


def test_get_dihedrals() -> None:
    """Tests for :func:`nanoCAT.ff.mol_topology.get_dihedrals`."""
    mol = MOL.copy()
    dihedrals_ref = np.load(join(PATH, 'get_dihedrals.npy'))
    dihedrals = get_dihedrals(mol)
    np.testing.assert_array_equal(dihedrals, dihedrals_ref)


def test_get_impropers() -> None:
    """Tests for :func:`nanoCAT.ff.mol_topology.get_impropers`."""
    mol = MOL.copy()
    impropers_ref = np.load(join(PATH, 'get_impropers.npy'))
    impropers = get_impropers(mol)
    np.testing.assert_array_equal(impropers, impropers_ref)
