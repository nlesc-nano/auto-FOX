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

from typing import Hashable, Iterable

import numpy as np
import pandas as pd

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
    df = pd.DataFrame(0.0, index=index, columns=atom_pairs)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Angstrom'
    return df


def get_rdf(dist: np.ndarray,
            dr: float = 0.05,
            r_max: float = 12.0) -> np.ndarray:
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
        Note that the RDF is summed (not averaged) w.r.t. the number of molecules.

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
    return dens.sum(axis=0)
