"""A module for testing :mod:`FOX.recipes.ligands`."""

from pathlib import Path

import numpy as np

from FOX import MultiMolecule, example_xyz
from FOX.recipes import get_lig_center

PATH = Path('tests') / 'test_files' / 'recipes'
MOL = MultiMolecule.from_xyz(example_xyz)


def test_get_lig_center() -> None:
    """Test :func:`get_lig_center`."""
    ref1 = np.load(PATH / 'get_lig_center1.npy')
    ref2 = np.load(PATH / 'get_lig_center2.npy')
    lig_center1 = get_lig_center(MOL, 123, 4, mass_weighted=True)
    lig_center2 = get_lig_center(MOL, 123, 4, mass_weighted=False)

    np.testing.assert_allclose(lig_center1, ref1)
    np.testing.assert_allclose(lig_center2, ref2)


def test_examples() -> None:
    """Test examples."""
    lig_centra1 = get_lig_center(MOL, 123, 4)
    lig_centra2 = lig_centra1[:, [0, 1, 2, 3]]

    mol_new1 = MOL.add_atoms(lig_centra1, symbols='Xx')
    mol_new2 = MOL.add_atoms(lig_centra2, symbols='Xx')

    rdf1 = mol_new1.init_rdf(atom_subset=['Xx'])
    rdf2 = mol_new2.init_rdf(atom_subset=['Xx'])
    adf1 = mol_new1.init_adf(atom_subset=['Xx'])

    ref1 = np.load(PATH / 'ligands_rdf1.npy')
    ref2 = np.load(PATH / 'ligands_rdf2.npy')
    ref3 = np.load(PATH / 'ligands_adf1.npy')

    np.testing.assert_allclose(rdf1, ref1)
    np.testing.assert_allclose(rdf2, ref2)
    np.testing.assert_allclose(adf1, ref3)
