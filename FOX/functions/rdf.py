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

from __future__ import annotations

from typing import Hashable, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ['get_rdf_df', 'get_rdf', 'get_rdf_lowmem']


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


def get_rdf(
    dist: np.ndarray,
    dr: float = 0.05,
    r_max: float = 12.0,
    *,
    volume: None | npt.ArrayLike = None,
) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    The RDF is calculated using the 3D distance matrix **dist**.

    Parameters
    ----------
    dist : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n, k)`
        A 3D array representing :math:`m` distance matrices of :math:`n` by :math:`k` atoms.
    dr : :class:`float`
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.
    r_max : :class:`float`
        The maximum to be evaluated interatomic distance.
    volume : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`()` or :math:`(m,)`, optional
        The volume containing the system in question; used for calculating the average density.
        If :data:`None`, estimate the volume by a sphere whose diameter is equal to
        the largest value in **dist**.

    Returns
    -------
    1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**:
        An array with the resulting radial distribution function.
        Note that the RDF is summed (not averaged) w.r.t. the number of molecules.

    """  # noqa: E501
    if not dist.size:
        return np.zeros((), dtype=dist.dtype)

    r = np.arange(0, r_max + dr, dr)
    idx_max = 1 + int(r_max / dr)
    int_step = 4 * np.pi * dr * r**2
    int_step[0] = np.nan

    dist_shape = dist.shape
    if volume is not None:
        dens_mean = dist_shape[2] / np.asarray(volume)
    else:
        dens_mean = dist_shape[2] / ((4/3) * np.pi * (0.5 * dist.max(axis=(1, 2)))**3)
    dist2 = (dist / dr).astype(np.int32)
    dist2.shape = dist_shape[0], dist_shape[1] * dist_shape[2]

    dens = np.array([np.bincount(i, minlength=idx_max)[:idx_max] for i in dist2], dtype=np.float64)
    if dens_mean.ndim == 0:
        denom = dist_shape[1] * int_step * dens_mean
    else:
        denom = dist_shape[1] * int_step * dens_mean[..., None]
    dens /= denom
    dens[:, 0] = 0.0
    return dens.sum(axis=0)


def get_rdf_lowmem(dist: np.ndarray,
                   dr: float = 0.05,
                   r_max: float = 12.0) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    The RDF is calculated using the 2D distance matrix **dist**.
    A more memory efficient implementation of :func:`FOX.functions.rdf.get_rdf`,
    which operates on a 3D distance matrix.

    Parameters
    ----------
    dist : :math:`n*k` |np.ndarray|_ [|np.float64|_]
        A 2D array representing a single distance matrices of :math:`n` by :math:`k` atoms.
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
    elif dist.ndim == 2:
        dist = dist.reshape((1,) + dist.shape)

    idx_max = 1 + int(r_max / dr)
    dist_int = np.array(dist / dr, dtype=int).ravel()

    # Calculate the average particle density N / V
    # The diameter of the spherical volume (V) is defined by the largest inter-particle distance
    dens_mean = dist.shape[2] / ((4/3) * np.pi * (0.5 * dist.max())**3)

    # Count the number of occurances of each (rounded) distance
    dens = np.bincount(dist_int, minlength=idx_max)[:idx_max]

    # Correct for the number of reference atoms
    dens = dens / dist.shape[1]
    dens[0] = np.nan

    # Convert the particle count into a partical density
    r = np.arange(0, r_max + dr, dr)
    dens /= (4 * np.pi * r**2 * dr)

    # Normalize and return the particle density
    return dens / dens_mean
