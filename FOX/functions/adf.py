"""A module for constructing angular distribution functions.

Index
-----
.. currentmodule:: FOX.functions.adf
.. autosummary::
    get_adf
    get_adf_df

API
---
.. autofunction:: get_adf
.. autofunction:: get_adf_df

"""

from __future__ import annotations

import sys
from typing import (
    Sequence,
    Hashable,
    Iterable,
    Callable,
    TypeVar,
    Tuple,
    List,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal
    from numpy import float64 as f8, int64 as i8

    _T = TypeVar("_T")
    _SCT = TypeVar("_SCT", bound=np.generic)

    _3Tuple = Tuple[_T, _T, _T]
    NDArray = np.ndarray[Any, np.dtype[_SCT]]

__all__ = ['get_adf_df', 'get_adf']


def _adf_inner_cdktree(
    m: NDArray[f8],
    n: int,
    r_max: float,
    idx_list: Iterable[_3Tuple[NDArray[np.bool_]]],
    lattice: None | NDArray[f8],
    periodicity: Iterable[Literal[0, 1, 2]] = range(3),
    weight: None | Callable[[NDArray[f8]], NDArray[f8]] = None,
) -> List[NDArray[f8]]:
    """Perform the loop of :meth:`.init_adf` with a distance cutoff."""
    # Construct slices and a distance matrix
    if lattice is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            dist, vec, idx = _adf_inner_cdktree_periodic(m, n, r_max, lattice, periodicity)
            ang: NDArray[f8] = np.arccos(np.einsum('jkl,jml->jkm', vec, vec))
            dist = np.maximum(dist[..., None], dist[..., None, :])
    else:
        tree = cKDTree(m)
        dist, idx = tree.query(m, n, distance_upper_bound=r_max, p=2)
        dist[dist == np.inf] = 0.0
        idx[idx == len(m)] = 0

        # Slice the Cartesian coordinates
        coords13: NDArray[f8] = m[idx]
        coords2: NDArray[f8] = m[..., None, :]

        # Construct (3D) angle- and distance-matrices
        with np.errstate(divide='ignore', invalid='ignore'):
            vec = ((coords13 - coords2) / dist[..., None])
            ang = np.arccos(np.einsum('jkl,jml->jkm', vec, vec))
            dist = np.maximum(dist[..., None], dist[..., None, :])
    ang[np.isnan(ang)] = 0.0

    # Radian (float) to degrees (int)
    ang_int: NDArray[i8] = np.degrees(ang).astype(np.int64)

    # Construct and return the ADF
    ret = []
    for i, j, k in idx_list:
        ijk: NDArray[np.bool_] = j[:, None, None] & i[idx][..., None] & k[idx][..., None, :]
        weights = weight(dist[ijk]) if weight is not None else None
        ret.append(get_adf(ang_int[ijk], weights=weights))
    return ret


def _adf_inner_cdktree_periodic(
    m: NDArray[f8],
    n: int,
    r_max: float,
    lattice: NDArray[f8],
    periodicity: Iterable[Literal[0, 1, 2]],
) -> Tuple[NDArray[f8], NDArray[f8], NDArray[np.intp]]:
    # Construct the (full) distance matrix and vectors
    dist, vec = _adf_inner_periodic(m, lattice, periodicity)

    # Apply `n` and `r_max`: truncate the number of distances/vectors
    idx1 = np.argsort(dist, axis=1)
    if n < idx1.shape[1]:
        idx1 = idx1[:, :n]
    dist = np.take_along_axis(dist, idx1, axis=1)
    mask = dist > r_max
    idx1[mask] = 0
    dist[mask] = 0.0

    # Return the subsets
    idx0 = np.empty_like(idx1)
    idx0[:] = np.arange(len(idx0))[..., None]
    i = idx0.ravel()
    j = idx1.ravel()
    vec_ret = vec[i, j].reshape(*dist.shape, 3)
    return dist, vec_ret, idx1


def _adf_inner(
    m: NDArray[f8],
    idx_list: Iterable[_3Tuple[NDArray[np.bool_]]],
    lattice: None | NDArray[f8],
    periodicity: Iterable[Literal[0, 1, 2]] = range(3),
    weight: None | Callable[[NDArray[f8]], NDArray[f8]] = None,
) -> List[NDArray[f8]]:
    """Perform the loop of :meth:`.init_adf` without a distance cutoff."""
    # Construct (3D) angle- and distance-matrices
    with np.errstate(divide='ignore', invalid='ignore'):
        if lattice is None:
            # Construct a distance matrix
            dist: NDArray[f8] = cdist(m, m)

            # Slice the Cartesian coordinates
            coords13: NDArray[f8] = m
            coords2: NDArray[f8] = m[..., None, :]

            vec: NDArray[f8] = (coords13 - coords2) / dist[..., None]
        else:
            dist, vec = _adf_inner_periodic(m, lattice, periodicity)
        ang: NDArray[f8] = np.arccos(np.einsum('jkl,jml->jkm', vec, vec))
        dist = np.maximum(dist[..., :, None], dist[..., None, :])
    ang[np.isnan(ang)] = 0.0

    # Radian (float) to degrees (int)
    ang_int: NDArray[i8] = np.degrees(ang).astype(np.int64)

    # Construct and return the ADF
    ret = []
    for i, j, k in idx_list:
        ijk: NDArray[np.bool_] = j[:, None, None] & i[..., None] & k[..., None, :]
        weights = weight(dist[ijk]) if weight is not None else None
        ret.append(get_adf(ang_int[ijk], weights=weights))
    return ret


def _adf_inner_periodic(
    m: NDArray[f8],
    lattice: NDArray[f8],
    periodicity: Iterable[Literal[0, 1, 2]],
) -> Tuple[NDArray[f8], NDArray[f8]]:
    """Construct the distance matrix and angle-defining vectors for periodic systems."""
    vec = m - m[..., None, :]
    lat_norm = np.linalg.norm(lattice, axis=-1)

    iterator = ((i, lat_norm[i]) for i in periodicity)
    for i, vec_len in iterator:
        vec[..., i][vec[..., i] > (vec_len / 2)] -= vec_len
        vec[..., i][vec[..., i] < -(vec_len / 2)] += vec_len

    dist = np.linalg.norm(vec, axis=-1)
    vec /= dist[..., None]
    return dist, vec


def get_adf_df(atom_pairs: Sequence[Hashable]) -> pd.DataFrame:
    """Construct and return a pandas dataframe filled to hold angular distribution functions.

    Parameters
    ----------
    atom_pairs : |Sequence|_ [|Hashable|_]
        A nested sequence of collumn names.

    Returns
    -------
    |pd.DataFrame|_:
        An empty dataframe.

    """
    # Create and return the DataFrame
    index = pd.RangeIndex(1, 181, name='phi  /  Degrees')
    df = pd.DataFrame(0.0, index=index, columns=atom_pairs)
    df.columns.name = 'Atom pairs'
    return df


def get_adf(
    ang: NDArray[np.integer[Any]],
    weights: None | NDArray[np.number[Any]] = None,
) -> NDArray[f8]:
    r"""Calculate and return the angular distribution function (ADF).

    Parameters
    ----------
    ang : |np.ndarray|_ [|np.int64|_]
        A 1D array of angles (:code:`dtype=int`) with all angles.
        Units should be in degrees.

    weights : |np.ndarray|_ [|np.float|_], optional
        A 1D array of weighting factors.
        Should be of the same length as **ang**.

    Returns
    -------
    :math:`m*180` |np.ndarray|_ [|np.float64|_]:
        A 1D array with an angular distribution function spanning all values between 0 and 180
        degrees.

    """
    # Calculate and normalize the density
    denominator = len(ang) / 180
    at_count: NDArray[i8] = np.bincount(ang, minlength=181)[1:181]
    dens: NDArray[f8] = at_count / denominator

    if weights is None:
        return dens

    # Weight (and re-normalize) the density based on the distance matrix **dist**
    area: f8 = dens.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        dens *= np.bincount(ang, weights=weights, minlength=181)[1:181] / at_count
        dens *= area / np.nansum(dens)
    dens[np.isnan(dens)] = 0.0
    return dens
