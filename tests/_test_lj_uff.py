"""A module for testing :mod:`FOX.functions.lj_uff`."""

from pathlib import Path
from itertools import combinations

import numpy as np
from assertionlib import assertion

from FOX.ff.lj_uff import UFF_DF, combine_sigma, combine_epsilon

PATH = Path('tests') / 'test_files'


def test_combine_sigma() -> None:
    """Test :func:`FOX.functions.lj_uff.combine_sigma`."""
    ref = np.load(PATH / 'uff_sigma.npy')
    sigma = np.array([combine_sigma(at1, at2)for at1, at2 in combinations(UFF_DF.index, r=2)])

    np.testing.assert_allclose(sigma, ref)
    assertion.assert_(combine_sigma, 'a', 'b', exception=ValueError)


def test_combine_epsilon() -> None:
    """Test :func:`FOX.functions.lj_uff.combine_epsilon`."""
    ref = np.load(PATH / 'uff_epsilon.npy')
    eps = np.array([combine_epsilon(at1, at2)for at1, at2 in combinations(UFF_DF.index, r=2)])

    np.testing.assert_allclose(eps, ref)
    assertion.assert_(combine_epsilon, 'a', 'b', exception=ValueError)
