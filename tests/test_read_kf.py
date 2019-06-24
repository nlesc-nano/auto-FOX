"""A module for testing files in the :mod:`FOX.io.read_kf` module."""

from os.path import join

import numpy as np

from FOX.io.read_kf import read_kf

__all__: list = []

REF_DIR = 'tests/test_files'


def test_read_kf():
    """Test :func:`FOX.io.read_psf.read_kf`."""
    ar, dict_ = read_kf(join(REF_DIR, 'md.rkf'))
    ref = np.load(join(REF_DIR, 'md_rkf.npy'))

    assert dict_ == {'C': [0, 1], 'H': [2, 3, 4, 5, 6, 7]}
    np.testing.assert_allclose(ar, ref)
