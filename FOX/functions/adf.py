"""A module for constructing angular distribution functions."""

from typing import (Sequence, Hashable, Dict)

import numpy as np
import pandas as pd

__all__ = ['get_adf']


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


def get_adf(ang: np.ndarray,
            dist: np.ndarray,
            atnum_ar: np.ndarray,
            atnum_dict: Dict[int, int],
            distance_weighted: bool = True) -> np.ndarray:
    r"""Calculate and return the angular distribution function (ADF).

    The ADF is based on the 3D angle matrix **ang_mat**.

    Parameters
    ----------
    ang : :math:`n*k*l` |np.ndarray|_ [|np.float64|_]
        A 3D angle matrix (radian) constructed from 1 molecule and three sets of
        :math:`n`, :math:`k` and :math:`l` atoms.

    dist : :math:`n*k*l` |np.ndarray|_ [|np.float64|_]
        A 3D distance matrix (Angstrom) containing all distances between three sets of
        :math:`n`, :math:`k` and :math:`l` atoms.

    distance_weighted : |bool|_
        Return the distance-weighted angular distribution function.
        Each angle, :math:`\phi_{ijk}`, is weighted by the distance according to the
        weighting factor :math:`v`:

        .. math::

            v = \Biggl \lbrace
            {
                e^{-r_{ji}}, \quad r_{ji} \; \gt \; r_{jk}
                \atop
                e^{-r_{jk}}, \quad r_{ji} \; \lt \; r_{jk}
            }

    Returns
    -------
    :math:`m*180` |np.ndarray|_ [|np.float64|_]:
        A 2D array with an angular distribution function spanning all values between 0 and 180
        degrees.

    """
    ang_int = np.degrees(ang).astype(dtype=int)

    ret = []
    for k, v in atnum_dict.items():
        # Prepare slices
        j = atnum_ar == k
        dist_flat = dist[j]

        # Calculate the average angle density
        r_max = -np.log(dist_flat.min())
        volume = (4/3) * np.pi * (0.5 * r_max)**3
        dens_mean = v / volume

        # Calculate and normalize the density
        denominator = dens_mean * (volume / 180) * len(dist_flat) / v
        at_count = np.array(np.bincount(ang_int[j], minlength=181)[1:181], dtype=float)
        dens = at_count / denominator

        # Weight (and re-normalize) the density based on the distance matrix **dist**
        if distance_weighted:
            area = dens.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                weight = np.bincount(ang_int[j], dist_flat, minlength=181)[1:181] / at_count
                dens *= weight
                normalize = area / np.nansum(dens)
                dens *= normalize
            dens[np.isnan(dens)] = 0.0
        ret.append(dens)

    return np.array(ret).T
