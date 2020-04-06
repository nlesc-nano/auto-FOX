"""A module for testing :mod:`FOX.armc.phi_updater`."""

import pandas as pd
import numpy as np

from assertionlib import assertion

from FOX.armc import PhiUpdater

PHI = PhiUpdater()


def test_call():
    """Test :meth:`PhiUpdater.__call__`."""
    values = [
        1,
        1.0,
        [1.0],
        np.array([1.0]),
        pd.Series([1.0])
    ]

    for v in values:
        scalar = np.atleast_1d(PHI(v))[0]
        assertion.eq(scalar, 2)


def test_update():
    """Test :meth:`PhiUpdater.update`."""
    phi = PHI.copy()

    acceptance = np.ones(100, dtype=bool)
    acceptance[75:] = False

    ref = phi.phi
    phi.update(acceptance)
    new = phi.phi

    assertion.eq(ref, 1)
    assertion.eq(new, 0.5)
