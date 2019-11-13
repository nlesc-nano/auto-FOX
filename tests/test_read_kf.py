"""A module for testing files in the :mod:`FOX.io.read_kf` module."""

from os.path import join

import numpy as np
from assertionlib import assertion

from FOX.io.read_kf import read_kf

PATH = join('tests', 'test_files')


def test_read_kf():
    """Test :func:`FOX.io.read_psf.read_kf`."""
    ar, dct = read_kf(join(PATH, 'md.rkf'))
    ref = np.load(join(PATH, 'md_rkf.npy'))

    assertion.eq(dct, {'C': [0, 1], 'H': [2, 3, 4, 5, 6, 7]})
    np.testing.assert_allclose(ar, ref)
