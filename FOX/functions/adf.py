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
    _T = TypeVar("_T")
    _SCT = TypeVar("_SCT", bound=np.generic)

    _3Tuple = Tuple[_T, _T, _T]
    NDArray = np.ndarray[Any, np.dtype[_SCT]]

__all__ = ['get_adf_df', 'get_adf']


def _adf_inner_cdktree(
    m: NDArray[np.float64],
    n: int,
    r_max: float,
    idx_list: Iterable[_3Tuple[NDArray[np.integer[Any]]]],
    boxsize: None | NDArray[np.float64],
    weight: None | Callable[[NDArray[np.float64]], NDArray[np.float64]] = None,
) -> List[NDArray[np.float64]]:
    """Perform the loop of :meth:`.init_adf` with a distance cutoff."""
    # Construct slices and a distance matrix
    tree = cKDTree(m, boxsize=boxsize)
    dist, idx = tree.query(m, n, distance_upper_bound=r_max, p=2)  # type: NDArray[np.float64], NDArray[np.intp]  # noqa: E501
    dist[dist == np.inf] = 0.0
    idx[idx == m.shape[0]] = 0

    # Slice the Cartesian coordinates
    coords13: NDArray[np.float64] = m[idx]
    coords2: NDArray[np.float64] = m[..., None, :]

    # Construct (3D) angle- and distance-matrices
    with np.errstate(divide='ignore', invalid='ignore'):
        vec: NDArray[np.float64] = ((coords13 - coords2) / dist[..., None])
        ang: NDArray[np.float64] = np.arccos(np.einsum('jkl,jml->jkm', vec, vec))
        dist = np.maximum(dist[..., None], dist[..., None, :])
    ang[np.isnan(ang)] = 0.0

    # Radian (float) to degrees (int)
    ang_int: NDArray[np.int64] = np.degrees(ang).astype(np.int64)

    # Construct and return the ADF
    ret = []
    for i, j, k in idx_list:
        ijk: NDArray[np.bool_] = j[:, None, None] & i[idx][..., None] & k[idx][..., None, :]
        weights = weight(dist[ijk]) if weight is not None else None
        ret.append(get_adf(ang_int[ijk], weights=weights))
    return ret


def _adf_inner(
    m: NDArray[np.float64],
    idx_list: Iterable[_3Tuple[NDArray[np.integer[Any]]]],
    weight: None | Callable[[NDArray[np.float64]], NDArray[np.float64]] = None,
) -> List[NDArray[np.float64]]:
    """Perform the loop of :meth:`.init_adf` without a distance cutoff."""
    # Construct a distance matrix
    dist: NDArray[np.float64] = cdist(m, m)

    # Slice the Cartesian coordinates
    coords13: NDArray[np.float64] = m
    coords2: NDArray[np.float64] = m[..., None, :]

    # Construct (3D) angle- and distance-matrices
    with np.errstate(divide='ignore', invalid='ignore'):
        vec: NDArray[np.float64] = ((coords13 - coords2) / dist[..., None])
        ang: NDArray[np.float64] = np.arccos(np.einsum('jkl,jml->jkm', vec, vec))
        dist = np.maximum(dist[..., None], dist[..., None, :])
    ang[np.isnan(ang)] = 0.0

    # Radian (float) to degrees (int)
    ang_int: NDArray[np.int64] = np.degrees(ang).astype(np.int64)

    # Construct and return the ADF
    ret = []
    for i, j, k in idx_list:
        ijk: NDArray[np.bool_] = j[:, None, None] & i[..., None] & k[..., None, :]
        weights = weight(dist[ijk]) if weight is not None else None
        ret.append(get_adf(ang_int[ijk], weights=weights))
    return ret


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
) -> NDArray[np.float64]:
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
    at_count: NDArray[np.int64] = np.bincount(ang, minlength=181)[1:181]
    dens: NDArray[np.float64] = at_count / denominator

    if weights is None:
        return dens

    # Weight (and re-normalize) the density based on the distance matrix **dist**
    area: np.float64 = dens.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        dens *= np.bincount(ang, weights=weights, minlength=181)[1:181] / at_count
        dens *= area / np.nansum(dens)
    dens[np.isnan(dens)] = 0.0
    return dens
