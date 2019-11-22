"""A module for testing the :class:`FOX.classes.armc.ARMC` class."""

from pathlib import Path

import numpy as np
from assertionlib import assertion

from FOX import ARMC

PATH: Path = Path('tests') / 'test_files'
ARMC_, _ = ARMC.from_yaml(PATH / 'armc.yaml')


def test_update_phi() -> None:
    """Test :meth:`ARMC.update_phi`."""
    armc = ARMC_.copy(deep=True)

    ref = armc.phi
    acceptance1 = np.array([True, True, True, True, False])
    acceptance2 = ~acceptance1

    armc.update_phi(acceptance1)
    assertion.eq(armc.phi, ref / 2)

    armc.update_phi(acceptance1)
    assertion.eq(armc.phi, ref / 4)

    armc.update_phi(acceptance2)
    armc.update_phi(acceptance2)
    assertion.eq(armc.phi, ref)


def test_apply_phi() -> None:
    """Test :attr:`ARMC.apply_phi`."""
    armc = ARMC_.copy(deep=True)

    assertion.is_(armc.apply_phi, np.add)


def test_super_iter_len() -> None:
    """Test :attr:`ARMC.super_iter_len`."""
    armc = ARMC_.copy(deep=True)

    assertion.eq(armc.super_iter_len, armc.iter_len // armc.sub_iter_len)
