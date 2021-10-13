"""A module for testing :mod:`FOX.recipes.param`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import numpy as np
from assertionlib import assertion
from FOX.recipes import get_best, overlay_descriptor, plot_descriptor

try:
    from matplotlib.image import imread
except ImportError as ex:
    MATPLOTLIB_EX: None | ImportError = ex
else:
    MATPLOTLIB_EX = None

PATH = Path('tests') / 'test_files' / 'recipes'
HDF5 = Path('tests') / 'test_files' / 'armc_test.hdf5'


def test_get_best() -> None:
    """Test :func:`FOX.recipes.param.get_best`."""
    keys = ('aux_error', 'aux_error_mod', 'param', 'rdf')

    for name in keys:
        ref = np.load(PATH / f'{name}.npy')
        value = get_best(HDF5, name=name)
        np.testing.assert_allclose(value, ref, err_msg=name)

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


@pytest.mark.skipif(MATPLOTLIB_EX is not None, reason="Requires matplotlib")
def test_plot_descriptor() -> None:
    """Test :func:`FOX.recipes.param.plot_descriptor`."""
    rdf = get_best(HDF5, name='rdf')
    rdf_dict = overlay_descriptor(HDF5, name='rdf')

    fig1 = plot_descriptor(rdf, show_fig=False)
    fig2 = plot_descriptor(rdf_dict, show_fig=False)

    name1 = str(PATH / 'tmp_fig1.png')
    name2 = str(PATH / 'tmp_fig2.png')

    ref1 = imread(str(PATH / 'ref1.png')).astype('int8')
    ref2 = imread(str(PATH / 'ref2.png')).astype('int8')

    try:
        fig1.savefig(name1, dpi=300, format='png')
        fig2.savefig(name2, dpi=300, format='png')

        ar1 = imread(name1).astype('int8')
        ar2 = imread(name2).astype('int8')

        # For reasons unclear np.testing.assert_allclose() does not work here
        assertion((np.abs(ar1 - ref1) <= 1).all())
        assertion((np.abs(ar2 - ref2) <= 1).all())
    finally:
        os.remove(name1) if os.path.isfile(name1) else None
        os.remove(name2) if os.path.isfile(name2) else None
