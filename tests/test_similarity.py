"""Tests for :mod:`FOX.recipes.similarity`."""

from __future__ import annotations
from typing import Mapping, Any, Type, TYPE_CHECKING
from pathlib import Path

import numpy as np
import h5py
import pytest
from FOX import MultiMolecule, example_xyz
from FOX.recipes import compare_trajectories

if TYPE_CHECKING:
    import numpy.typing as npt

MOL1 = MultiMolecule.from_xyz(example_xyz)[:100]
MOL2 = MOL1 * 1.25
HDF5_FILE = Path('tests') / 'test_files' / 'test_similarity.hdf5'

with h5py.File(HDF5_FILE, 'r') as f:
    REF: np.ndarray = f['cosine'][:]
    REF.setflags(write=False)


def _sqeuclidean(md: np.ndarray, md_ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(md - md_ref, axis=-1)**2


def rmsd(a: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(a**2, axis=axis)**0.5


class TestCompareTrajectories:
    """Tests for :func:`FOX.recipes.compare_trajectories`."""

    @pytest.mark.parametrize(
        'name,kwargs',
        [
            ('cosine', {'metric': 'cosine'}),
            ('euclidean', {'metric': 'euclidean'}),
            ('euclidean_no_reset', {'metric': 'euclidean', 'reset_origin': False}),
            ('euclidean_p1', {'metric': 'euclidean', 'p': 1}),
            ('sqeuclidean', {'metric': _sqeuclidean}),
            ('sum', {'reduce': lambda n: np.sum(n, axis=-1)}),
            ('rmsd', {'reduce': lambda n: rmsd(n, axis=-1)}),
            ('no_reduce', {'reduce': None}),
        ]
    )
    def test_succeeds(self, name: str, kwargs: Mapping[str, Any]) -> None:
        """Tests for succesful :func:`~FOX.recipes.compare_trajectories` calls."""
        with h5py.File(HDF5_FILE, 'r') as f:
            ref = f[name][:]

        out = compare_trajectories(MOL1, MOL2, **kwargs)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-8)

    @pytest.mark.parametrize('md', [MOL1, MOL1.view(np.ndarray), MOL1.tolist()])
    @pytest.mark.parametrize('md_ref', [MOL2, MOL2.view(np.ndarray), MOL2.tolist()])
    def test_array_like(self, md: npt.ArrayLike, md_ref: npt.ArrayLike,) -> None:
        """Tests :func:`~FOX.recipes.compare_trajectories` with array-like objects."""
        out = compare_trajectories(md, md_ref)
        np.testing.assert_allclose(out, REF, rtol=0, atol=1e-8)

    @pytest.mark.parametrize('md', [MOL1, MOL1[0]])
    @pytest.mark.parametrize('md_ref', [MOL2, MOL2[0]])
    def test_ndim(self, md: npt.ArrayLike, md_ref: npt.ArrayLike,) -> None:
        """Tests :func:`~FOX.recipes.compare_trajectories` with array-like objects."""
        _ = compare_trajectories(md, md_ref)

    @pytest.mark.parametrize(
        'name,exc_type,kwargs',
        [
            ('dtype', TypeError, {'md': [[[object()]]], 'md_ref': MOL2}),
            ('dtype', TypeError, {'md': MOL1, 'md_ref': [[[object()]]]}),
            ('metric', ValueError, {'md': MOL1, 'md_ref': MOL2, 'metric': 'bob'}),
            ('metric', TypeError, {'md': MOL1, 'md_ref': MOL2, 'metric': 1}),
            ('metric', TypeError, {'md': MOL1, 'md_ref': MOL2, 'metric': [1]}),
        ]
    )
    def test_raises(self, name: str, exc_type: Type[Exception], kwargs: Mapping[str, Any]) -> None:
        """Tests for :func:`~FOX.recipes.compare_trajectories` failures."""
        with pytest.raises(exc_type):
            compare_trajectories(**kwargs)
