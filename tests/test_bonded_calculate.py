"""A module for testing :mod:`FOX.ff.bonded_calculate`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest
import numpy as np
from assertionlib import assertion

from FOX import MultiMolecule, get_bonded

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import float64 as f8
    import _pytest

PATH = Path('tests') / 'test_files'


class OutputTuple(NamedTuple):
    bonds: None | NDArray[f8]
    angles: None | NDArray[f8]
    urey_bradley: None | NDArray[f8]
    dihedrals: None | NDArray[f8]
    impropers: None | NDArray[f8]


class TestGetBonded:
    """Test :func:`FOX.functions.lj_calculate.get_non_bonded`."""

    @pytest.fixture(scope="class", autouse=True)
    def output(self, request: _pytest.fixtures.SubRequest) -> OutputTuple:
        """Generate the test output of :func:`fast_bulk_workflow`."""
        psf = PATH / 'Cd68Se55_26COO_MD_trajec.psf'
        prm = PATH / 'Cd68Se55_26COO_MD_trajec.prm'
        mol = MultiMolecule.from_xyz(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')
        ret = get_bonded(mol, psf, prm)
        return OutputTuple(*ret)

    def test_bonds(self, output: OutputTuple) -> None:
        ref = np.load(PATH / 'get_bonded.bonds.npy')
        assert output.bonds is not None
        np.testing.assert_allclose(output.bonds, ref)

    def test_angles(self, output: OutputTuple) -> None:
        ref = np.load(PATH / 'get_bonded.angles.npy')
        assert output.angles is not None
        np.testing.assert_allclose(output.angles, ref)

    def test_urey_bradley(self, output: OutputTuple) -> None:
        ref = np.load(PATH / 'get_bonded.urey_bradley.npy')
        assert output.urey_bradley is not None
        np.testing.assert_allclose(output.urey_bradley, ref)

    def test_dihedrals(self, output: OutputTuple) -> None:
        assertion.is_(output.dihedrals, None)

    def test_impropers(self, output: OutputTuple) -> None:
        ref = np.load(PATH / 'get_bonded.impropers.npy')
        assert output.impropers is not None
        np.testing.assert_allclose(output.impropers, ref)
