"""A module for constructing angular distribution functions."""

from typing import (Sequence, Hashable)

import numpy as np
import pandas as pd

__all__ = ['get_adf']


def get_adf_df(atom_pairs: Sequence[Sequence[Hashable]]) -> pd.DataFrame:
    """Construct and return a pandas dataframe filled to hold angular distribution functions.

    Parameters
    ----------
    dict atom_pairs:
        A dictionary of 3-tuples representing the keys of the dataframe.

    Returns
    -------
    |pd.DataFrame|_:
        An empty dataframe.
    """
    # Prepare the DataFrame arguments
    shape = 181, len(atom_pairs)
    index = np.arange(181)

    # Create and return the DataFrame
    df = pd.DataFrame(np.empty(shape), index=index, columns=atom_pairs)
    df.columns.name = 'Atom pairs'
    df.index.name = 'phi  /  Degrees'
    return df


def get_adf(ang_mat: np.ndarray,
            r_max: float = 24.0) -> np.ndarray:
    """Calculate and return the angular distribution function (ADF).

    The ADF is based on the 4D angle matrix **ang_mat**.

    Parameters
    ----------
    |np.ndarray|_ ang:
        A 4D angle matrix constructed from :math:`m` molecules and three sets of
        :math:`n`, :math:`k` and :math:`l` atoms.

    float r_max:
        The diameter of the sphere used for converting particle counts into densities.

    Returns
    -------
    :math:`181` |np.ndarray|_ [|np.float64|_]:
        A 1D array with an angular distribution function spanning all values between 0 and 180
        degrees.
    """
    ang_mat[np.isnan(ang_mat)] = 10
    ang_int = np.array(np.degrees(ang_mat), dtype=int)
    volume = (4/3) * np.pi * (0.5 * r_max)**3
    dens_mean = ang_mat.shape[1] / volume

    dens = np.array([np.bincount(i.flatten(), minlength=181)[:181] for i in ang_int], dtype=float)
    dens /= np.product(ang_mat.shape[-2:])  # Correct for the number of reference atoms
    dens /= volume / 181  # Convert the particle count into a partical density
    dens /= dens_mean  # Normalize the particle density
    return np.average(dens, axis=0)
