"""A module for testing :mod:`FOX.functions.lj_calculate`."""

import os
from pathlib import Path

import numpy as np
from matplotlib.image import imread
from assertionlib import assertion

from FOX.recipes import get_best, overlay_descriptor, plot_descriptor

PATH = Path('tests') / 'test_files' / 'recipes'
PATH = Path('/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files/recipes')

HDF5 = Path('tests') / 'test_files' / 'armc_test.hdf5'
HDF5 = Path('/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files/armc_test.hdf5')


def test_get_best() -> None:
    """Test :func:`FOX.recipes.param.get_best`."""
    keys = ('aux_error', 'aux_error_mod', 'param', 'phi', 'rdf')

    for name in keys:
        ref = np.load(PATH / f'{name}.npy')
        value = get_best(HDF5, name=name)
        try:
            np.testing.assert_allclose(value.values, ref)
        except AttributeError:  # value is a float
            np.testing.assert_allclose(value, ref)

    assertion.assert_(get_best, HDF5, name='bob', exception=KeyError)
    assertion.assert_(get_best, HDF5, name='rdf', i=1, exception=KeyError)


def test_overlay_descriptor() -> None:
    """Test :func:`FOX.recipes.param.overlay_descriptor`."""
    # Required for Python <= 3.6, as dictionaries are not necessarily ordered prior to 3.7
    idx_map = {'Cd Cd': 0, 'Cd Se': 1, 'Cd O': 2, 'Se Se': 3, 'Se O': 4, 'O O': 5}

    rdf_dict = overlay_descriptor(HDF5, name='rdf')
    ref_ar = np.load(PATH / 'overlay_descriptor.npy')

    for k, i in idx_map.items():
        rdf, ref = rdf_dict[k], ref_ar[i]
        np.testing.assert_allclose(rdf, ref)


def test_plot_descriptor() -> None:
    """Test :func:`FOX.recipes.param.plot_descriptor`."""
    rdf = get_best(HDF5, name='rdf')
    rdf_dict = overlay_descriptor(HDF5, name='rdf')

    fig1 = plot_descriptor(rdf)
    fig2 = plot_descriptor(rdf_dict)

    name1 = str(PATH / 'tmp_fig1.png')
    name2 = str(PATH / 'tmp_fig2.png')

    ref1 = np.array(imread(str(PATH / 'ref1.png')))
    ref2 = np.array(imread(str(PATH / 'ref2.png')))

    try:
        fig1.savefig(name1, dpi=300, quality=100, format='png')
        fig2.savefig(name2, dpi=300, quality=100, format='png')

        ar1 = np.array(imread(name1))
        ar2 = np.array(imread(name2))

        np.testing.assert_allclose(ar1, ref1)
        np.testing.assert_allclose(ar2, ref2)
    finally:
        os.remove(name1) if os.path.isfile(name1) else None
        os.remove(name2) if os.path.isfile(name2) else None
