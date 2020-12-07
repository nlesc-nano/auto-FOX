"""A module for testing time-resolved distribution functions."""

from pathlib import Path

import numpy as np
from FOX import MultiMolecule, example_xyz
from FOX.recipes import time_resolved_rdf, time_resolved_adf

MOL = MultiMolecule.from_xyz(example_xyz)
PATH = Path('tests') / 'test_files'


def test_td_rdf():
    """Test :func:`FOX.recipes.time_resolved_rdf`."""
    mol = MOL.copy()
    rdf_list = time_resolved_rdf(mol, atom_subset=['Cd', 'Se'])

    array = np.array([i.values for i in rdf_list])
    array_ref = np.load(PATH / 'time_resolved_rdf.npy')
    np.testing.assert_allclose(array, array_ref)


def test_td_adf():
    """Test :func:`FOX.recipes.time_resolved_adf`."""
    mol = MOL.copy()
    adf_list = time_resolved_adf(mol, atom_subset=['Cd', 'Se'], r_max=6.0)

    array = np.array([i.values for i in adf_list])
    array_ref = np.load(PATH / 'time_resolved_adf.npy')
    np.testing.assert_allclose(array, array_ref)
