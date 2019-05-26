"""A module for constructing radial distribution functions."""

import numpy as np
import pandas as pd

__all__ = ['get_rdf_lowmem', 'get_rdf']


def get_rdf_df(atom_pairs,
               dr: float = 0.05,
               r_max: float = 12.0) -> pd.DataFrame:
    """Construct and return a pandas dataframe filled with zeros.

    Parameters
    ----------
    dict atom_pairs:
        Aa dictionary of 2-tuples representing the keys of the dataframe.

    float dr:
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    float r_max:
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    |pd.DataFrame|_:
        An empty dataframe to hold the RDF.

    """
    # Prepare the DataFrame arguments
    shape = 1 + int(r_max / dr), len(atom_pairs)
    index = np.arange(0, r_max + dr, dr)

    # Create and return the DataFrame
    df = pd.DataFrame(np.zeros(shape), index=index, columns=atom_pairs)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Ångström'
    return df


def get_rdf(dist: np.ndarray,
            dr: float = 0.05,
            r_max: float = 12.0) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    The RDF is calculated using the 3D distance matrix **dist**.

    Parameters
    ----------
    |np.ndarray|_ dist:
        A 3D array representing :math:`m` distance matrices of :math:`n` by :math:`k` atoms.

    float dr:
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    float r_max:
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**:
        An array with the resulting radial distribution function.

    """
    r = np.arange(0, r_max + dr, dr)
    idx_max = 1 + int(r_max / dr)
    int_step = 4 * np.pi * dr * r**2
    int_step[0] = np.nan

    dist_shape = dist.shape
    dens_mean = dist_shape[2] / ((4/3) * np.pi * (0.5 * dist.max(axis=(1, 2)))**3)
    dist2 = dist / dr
    dist2 = dist2.astype(np.int32, copy=False)
    dist2.shape = dist_shape[0], dist_shape[1] * dist_shape[2]

    dens = np.array([np.bincount(i, minlength=idx_max)[:idx_max] for i in dist2], dtype=float)
    dens /= dist_shape[1]
    dens /= int_step
    dens /= dens_mean[:, None]
    dens[:, 0] = 0.0
    return np.average(dens, axis=0)


def get_rdf_lowmem(dist: np.ndarray,
                   dr: float = 0.05,
                   r_max: float = 12.0) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    The RDF is calculated using the 2D distance matrix **dist**.

    A more memory efficient implementation of :func:`FOX.functions.rdf.get_rdf`,
    which operates on a 3D distance matrix.

    Parameters
    ----------
    |np.ndarray|_ dist:
        A 2D array representing a single distance matrices of :math:`n` by :math:`k` atoms.

    float dr:
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    float r_max:
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**:
        An array with the resulting radial distribution function.

    """
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
