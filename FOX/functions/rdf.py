"""A module for constructing radial distribution functions.

Index
-----
.. currentmodule:: FOX.functions.adf
.. autosummary::
    get_rdf
    get_rdf_df

API
---
.. autofunction:: get_rdf
.. autofunction:: get_rdf_df

"""

from typing import Hashable, Iterable, Optional, Mapping, Any, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import ckdtree

from ..utils import slice_iter

__all__ = ['get_rdf_df', 'get_rdf']


def get_rdf_df(atom_pairs: Iterable[Hashable],
               dr: float = 0.05,
               r_max: float = 12.0) -> pd.DataFrame:
    """Construct and return a pandas dataframe filled with zeros.

    Parameters
    ----------
    atom_pairs : dict
        Aa dictionary of 2-tuples representing the keys of the dataframe.

    dr : float
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    r_max : float
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    |pd.DataFrame|_:
        An empty dataframe to hold the RDF.

    """
    # Create and return the DataFrame
    index = np.arange(0.0, r_max + dr, dr)
    df = pd.DataFrame(0, index=index, columns=atom_pairs, dtype=np.float64)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Angstrom'
    return df


def _init_rdf(
    coords: np.ndarray,
    df: pd.DataFrame,
    atom_pairs: Mapping[Any, Tuple[np.ndarray, np.ndarray]],
    r_max: float = 10.0,
    dr: float = 0.05,
    k: Optional[int] = None,
    nbytes_max: int = 1024**3,
) -> None:
    for name, (idx0, idx1) in atom_pairs.items():
        # Use either the full distance matrix or
        # a truncated one with up to `k` distances per atom
        if k is None:
            dist_mat_size = len(idx0) * len(idx1)
            func = lambda slc: np.array(
                [cdist(i, j) for i, j in zip(coords[slc, idx0], coords[slc, idx1])]
            )
        else:
            tree = ckdtree(coords[0, idx1])
            idx, dist = tree.query(coords[0, idx0], k, distance_upper_bound=r_max)
            is_fin = dist.ravel() != np.inf

            _idx0 = np.empty_like(idx)
            _idx0[:] = np.arange(len(idx))
            idx0 = _idx0.ravel()[is_fin]
            idx1 = idx.ravel()[is_fin]

            dist_mat_size = len(idx0)
            func = lambda slc: np.linalg.norm(coords[slc, idx0] - coords[slc, idx1], axis=-1)

        # Limit the size of the distance matrix to **nbytes_max**
        n = len(coords)
        dist_size = coords.dtype.itemsize * n * dist_mat_size
        n_step = np.ceil(n / (dist_size / nbytes_max)).astype(np.int64)

        iterator = slice_iter(n=n, n_step=n_step)
        for slc in iterator:
            dist = func(slc)
            df[name] += get_rdf(dist, dr, r_max)
    df /= len(coords)
    return


def get_rdf(dist: np.ndarray,
            dr: float = 0.05,
            r_max: float = 10.0) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    The RDF is calculated using the 3D distance matrix **dist**.

    Parameters
    ----------
    dist : :math:`m*n*k` |np.ndarray|_ [|np.float64|_]
        A 3D array representing :math:`m` distance matrices of :math:`n` by :math:`k` atoms.

    dr : float
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    r_max : float
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**:
        An array with the resulting radial distribution function.

    """
    if not dist.size:
        return np.zeros((), dtype=dist.dtype)

    r = np.arange(0, r_max + dr, dr)
    idx_max = 1 + int(r_max / dr)
    int_step = 4 * np.pi * dr * r**2
    int_step[0] = np.nan

    dist_shape = dist.shape
    dens_mean = dist_shape[2] / ((4/3) * np.pi * (0.5 * dist.max(axis=(1, 2)))**3)
    dist2 = (dist / dr).astype(np.int32)
    dist2.shape = dist_shape[0], dist_shape[1] * dist_shape[2]

    dens = np.array([np.bincount(i, minlength=idx_max)[:idx_max] for i in dist2], dtype=float)
    denom = dist_shape[1] * int_step * dens_mean[:, None]
    dens /= denom
    dens[:, 0] = 0.0
    return dens
