"""A module for constructing angular distribution functions."""

from typing import Sequence, Hashable, Optional

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


def get_adf(ang: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
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
    at_count = np.bincount(ang, minlength=181)[1:181]
    dens = at_count / denominator

    if weights is None:
        return dens

    # Weight (and re-normalize) the density based on the distance matrix **dist**
    area = dens.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        dens *= np.bincount(ang, weights=weights, minlength=181)[1:181] / at_count
        dens *= area / np.nansum(dens)
    dens[np.isnan(dens)] = 0.0
    return dens
