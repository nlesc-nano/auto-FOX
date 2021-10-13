"""Tests for :mod:`FOX.recipes.similarity`."""

from __future__ import annotations\

import operator
from functools import partial
from typing import Mapping, Any, Type, Sequence, TYPE_CHECKING
from pathlib import Path

import numpy as np
import h5py
import pytest
from scipy.spatial.distance import cdist
from nanoutils import SetAttr

import FOX
from FOX import MultiMolecule, example_xyz
from FOX.recipes import compare_trajectories, fps_reduce
from FOX.recipes.similarity import CAT_EX

if TYPE_CHECKING:
    import numpy.typing as npt

MOL1 = MultiMolecule.from_xyz(example_xyz)[:100]
MOL2 = MOL1 * 1.25
HDF5_FILE = Path('tests') / 'test_files' / 'test_similarity.hdf5'
DIST = cdist(MOL1[0], MOL2[0])

with h5py.File(HDF5_FILE, 'r') as f:
    REF: np.ndarray = f['cosine'][:]
    REF.setflags(write=False)


def _sqeuclidean(md: np.ndarray, md_ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(md - md_ref, axis=-1)**2


def rmsd(a: np.ndarray, axis: int | Sequence[int]) -> np.ndarray:
    return np.mean(a**2, axis=axis)**0.5


@pytest.mark.skipif(CAT_EX is not None, reason="Requires CAT")
class TestCompareTrajectories:
    """Tests for :func:`FOX.recipes.compare_trajectories`."""

    @pytest.mark.parametrize(
        'name,kwargs',
        [
            ('cosine', {'metric': 'cosine'}),
            ('euclidean', {'metric': 'euclidean'}),
            ('euclidean_no_reset', {'metric': 'euclidean', 'reset_origin': False}),
            ('euclidean_p1', {'metric': 'minkowski', 'p': 1}),
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

    @pytest.mark.parametrize(
        'name,exc_type,kwargs',
        [
            ('dtype', TypeError, {'md': [[[object()]]], 'md_ref': MOL2}),
            ('dtype', TypeError, {'md': MOL1, 'md_ref': [[[object()]]]}),
            ('metric', ValueError, {'md': MOL1, 'md_ref': MOL2, 'metric': 'bob'}),
            ('metric', TypeError, {'md': MOL1, 'md_ref': MOL2, 'metric': 1}),
            ('metric', TypeError, {'md': MOL1, 'md_ref': MOL2, 'metric': [1]}),
            ('ndim', ValueError, {'md': MOL1[None], 'md_ref': MOL2}),
            ('ndim', ValueError, {'md': MOL1, 'md_ref': MOL2[None]}),
            ('len', ValueError, {'md': MOL1[:10], 'md_ref': MOL2}),
            ('len', ValueError, {'md': MOL1, 'md_ref': MOL2[:10]}),
        ]
    )
    def test_raises(self, name: str, exc_type: Type[Exception], kwargs: Mapping[str, Any]) -> None:
        """Tests for :func:`~FOX.recipes.compare_trajectories` failures."""
        with pytest.raises(exc_type):
            compare_trajectories(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n": 1},
            {"n": 20},
            {"n": 20, "operation": "min"},
            {"n": 20, "cluster_size": 5},
            {"n": 20, "cluster_size": [1, 1, 1, 1, 2, 2, 4]},
            {"n": 20, "start": 5},
            {"n": 20, "weight": partial(operator.__truediv__, 1)},
        ]
    )
    def test_fps(self, kwargs: Mapping[str, Any]) -> None:
        """Tests for succesful :func:`~FOX.recipes.fps_reduce` calls."""
        name = f'test_fps-{kwargs}'
        with h5py.File(HDF5_FILE, 'r+') as f:
            ref = f[name][:]

        out = fps_reduce(DIST, **kwargs)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-8)

    @pytest.mark.parametrize(
        "name,value",
        [
            ("0d", np.array(1)),
            ("1d", np.array(1, ndmin=1)),
            ("3d", np.array(1, ndmin=3)),
        ]
    )
    def test_fps_raises(self, name: str, value: np.ndarray) -> None:
        """Tests for :func:`~FOX.recipes.fps_reduce` failures."""
        with pytest.raises(ValueError):
            fps_reduce(value)

    def test_fps_no_cat(self) -> None:
        """Tests for :func:`~FOX.recipes.fps_reduce` calls without :mod:`CAT`."""
        with SetAttr(FOX.recipes.similarity, "CAT_EX", ImportError()), pytest.raises(ImportError):
            fps_reduce(DIST)
